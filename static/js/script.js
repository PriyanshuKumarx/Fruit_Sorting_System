document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files.length > 0) {
                fileInfo.textContent = this.files[0].name;
                fileInfo.classList.add('has-file');
            } else {
                fileInfo.textContent = 'No file selected';
                fileInfo.classList.remove('has-file');
            }
        });
    }

    $('.custom-file-input').on('change', function() {
        let fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').addClass("selected").html(fileName);
    });


    window.useSampleImage = function(filename) {
        alert("Sample image selected: " + filename + "\n\nIn a full implementation, this would automatically load and analyze the sample image.");
        
    };

    const uploadWrapper = document.querySelector('.upload-wrapper');
    if (uploadWrapper) {
        uploadWrapper.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });

        uploadWrapper.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });

        uploadWrapper.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                fileInfo.textContent = e.dataTransfer.files[0].name;
                fileInfo.classList.add('has-file');

                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        });
    }

    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.card, .feature-box, .jumbotron');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, {threshold: 0.1});

        elements.forEach(element => {
            observer.observe(element);
        });
    };

    animateOnScroll();
    document.querySelector('.btn-print')?.addEventListener('click', function() {
        window.print();
    });

    const uploadForm = document.querySelector('.upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            document.getElementById('loadingOverlay').classList.add('active');
        });
    }
});

async function analyzeViaAPI(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {
        document.getElementById('loadingOverlay')?.classList.add('active');
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        // Hide loading state
        document.getElementById('loadingOverlay')?.classList.remove('active');
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        document.getElementById('loadingOverlay')?.classList.remove('active');
        return null;
    }
}