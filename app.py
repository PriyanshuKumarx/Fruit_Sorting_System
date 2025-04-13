import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                             precision_recall_curve, roc_curve, auc)
from flask import Flask, render_template, request, redirect, url_for, jsonify
import uuid
import json
from datetime import datetime
from io import BytesIO
import base64
from collections import Counter


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  

DATASET_PATH = os.path.join('dataset', 'images')
MODEL_PATH = 'fruit_quality_model.h5'
METRICS_FOLDER = 'static/metrics'

NUTRITION_FACTS = {
    'apple': {'calories': 52, 'carbs': 14, 'fiber': 2.4, 'vitamin_c': '8% DV'},
    'banana': {'calories': 89, 'carbs': 23, 'fiber': 2.6, 'vitamin_c': '14% DV'},
    'orange': {'calories': 47, 'carbs': 12, 'fiber': 2.4, 'vitamin_c': '88% DV'}
}

STORAGE_TIPS = {
    'fresh': {
        'apple': 'Store in cool, humid place (32-35°F) for longest shelf life',
        'banana': 'Store at room temperature, away from other fruits',
        'orange': 'Store at room temperature or in refrigerator'
    },
    'rotten': {
        'apple': 'Compost or discard - not suitable for consumption',
        'banana': 'Can be used for baking if not moldy',
        'orange': 'Discard if moldy or fermented smell present'
    }
}

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(METRICS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join('static', 'misclassified', 'avocados'), exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_data(data_path, img_size=100):
    """Load and preprocess the dataset"""
    X, y, labels = [], [], []
    class_map = {}
    class_id = 0

    for fruit in os.listdir(data_path):
        fruit_folder = os.path.join(data_path, fruit)
        if not os.path.isdir(fruit_folder): continue

        for quality in os.listdir(fruit_folder):
            quality_folder = os.path.join(fruit_folder, quality)
            if not os.path.isdir(quality_folder): continue

            label_name = f"{fruit.split()[0]}_{quality}"
            if label_name not in class_map:
                class_map[label_name] = class_id
                labels.append(label_name)
                class_id += 1

            for img_file in os.listdir(quality_folder):
                img_path = os.path.join(quality_folder, img_file)
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(class_map[label_name])

    if len(X) == 0:
        raise ValueError("No images loaded. Please check dataset path and contents.")

    X = np.array(X) / 255.0
    y = np.array(y)
    y = to_categorical(y, num_classes=len(class_map))
    return X, y, labels, class_map

def build_model(input_shape, num_classes):
    """Build the CNN model architecture"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def generate_metrics(model, X_test, y_test, class_labels):
    """Generate and save evaluation metrics"""
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_true_classes, y_pred_classes), 
                annot=True, xticklabels=class_labels, 
                yticklabels=class_labels, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()

    labels_for_report = list(range(len(class_labels)))
    report = classification_report(y_true_classes, y_pred_classes, 
                                  target_names=class_labels, 
                                  labels=labels_for_report, output_dict=True)
    with open(os.path.join(METRICS_FOLDER, 'classification_report.json'), 'w') as f:
        json.dump(report, f)

    y_true_binary = np.array(['rotten' in class_labels[i] for i in y_true_classes])
    y_pred_binary = y_pred[:, [i for i, label in enumerate(class_labels) if 'rotten' in label]].sum(axis=1)
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Fresh vs Rotten)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(METRICS_FOLDER, 'roc_curve.png'))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(METRICS_FOLDER, 'precision_recall_curve.png'))
    plt.close()

    if hasattr(model, 'history'):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(model.history.history['accuracy'], label='Training Accuracy')
        plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(model.history.history['loss'], label='Training Loss')
        plt.plot(model.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.savefig(os.path.join(METRICS_FOLDER, 'training_history.png'))
        plt.close()
    class_counts = Counter(np.argmax(y_test, axis=1))
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_labels, y=[class_counts[i] for i in range(len(class_labels))])
    plt.title('Class Distribution in Test Set')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.savefig(os.path.join(METRICS_FOLDER, 'class_distribution.png'))
    plt.close()


    metadata = {
        'accuracy': float(acc),
        'loss': float(loss),
        'classes': class_labels,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'roc_auc': float(roc_auc)
    }
    with open(os.path.join(METRICS_FOLDER, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f)

    return metadata, report

def train_model():
    """Train the fruit quality classification model"""
    print("🚀 Starting model training...")

    X, y, class_labels, class_map = load_data(DATASET_PATH)
    print("🔍 Classes found:", class_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)


    model = build_model((100, 100, 3), len(class_labels))

    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    train_generator = datagen.flow(X_train_final, y_train_final, batch_size=32)
    history = model.fit(train_generator, epochs=30, validation_data=(X_val, y_val))

    metadata, report = generate_metrics(model, X_test, y_test, class_labels)

    save_model(model, MODEL_PATH)
    print("💾 Model saved successfully!")

    return model, class_labels, metadata, report

def load_existing_model():
    """Load a pre-trained model if available"""
    print("🔍 Loading existing model...")
    model = load_model(MODEL_PATH)
    
    with open(os.path.join(METRICS_FOLDER, 'model_metadata.json')) as f:
        metadata = json.load(f)
        class_labels = metadata['classes']

    with open(os.path.join(METRICS_FOLDER, 'classification_report.json')) as f:
        report = json.load(f)

    return model, class_labels, metadata, report


@app.route('/')
def home():
    """Render the main page"""
    try:
        with open(os.path.join(METRICS_FOLDER, 'model_metadata.json')) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {
            'accuracy': 0,
            'loss': 0,
            'classes': [],
            'training_date': 'Not trained yet',
            'roc_auc': 0
        }
    return render_template('index.html', metadata=metadata)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):

        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
 
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (100, 100)) / 255.0
        img_array = np.expand_dims(img_resized, axis=0)
        
   
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        fruit_type, quality = predicted_class.split('_')
        quality_status = 'Fresh' if 'fresh' in quality.lower() else 'Rotten'
        fruit_type_lower = fruit_type.lower()
        plt.figure(figsize=(8, 4))
        sns.barplot(x=class_labels, y=prediction[0])
        plt.title('Prediction Probabilities')
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        plt.tight_layout()

        prob_plot = BytesIO()
        plt.savefig(prob_plot, format='png')
        prob_plot.seek(0)
        prob_plot_url = base64.b64encode(prob_plot.getvalue()).decode('utf-8')
        plt.close()

        nutrition = NUTRITION_FACTS.get(fruit_type_lower, {})
        storage_tip = STORAGE_TIPS.get(quality.lower(), {}).get(fruit_type_lower, 
                                'No specific storage tips available')

        return render_template('result.html',
                             image_path=filename,
                             fruit_type=fruit_type.capitalize(),
                             quality=quality_status,
                             confidence=f"{confidence:.2f}%",
                             class_labels=class_labels,
                             classification_report=report,
                             prob_plot_url=prob_plot_url,
                             nutrition=nutrition,
                             storage_tip=storage_tip,
                             predicted_class=predicted_class,
                             metadata=metadata)
    
    return redirect(request.url)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if file and allowed_file(file.filename):
        # Process image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (100, 100)) / 255.0
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(np.max(prediction)),
            'probabilities': {cls: float(prob) for cls, prob in zip(class_labels, prediction[0])}
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Initialize directories
    create_directories()

    # Load or train model
    if os.path.exists(MODEL_PATH):
        model, class_labels, metadata, report = load_existing_model()
    else:
        model, class_labels, metadata, report = train_model()

    # Run the app
    app.run(debug=True)