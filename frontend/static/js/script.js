// frontend/static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileNameSpan = document.getElementById('file-name');
    const generateBtn = document.getElementById('generate-btn');
    const loader = document.getElementById('loader');
    const resultsDashboard = document.getElementById('results-dashboard');

    fileInput.addEventListener('change', () => {
        fileNameSpan.textContent = fileInput.files.length > 0 ? fileInput.files[0].name : 'No file chosen';
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please select an image file.');
            return;
        }

        // Show loader and hide old results
        loader.style.display = 'block';
        resultsDashboard.classList.add('hidden');
        generateBtn.disabled = true;
        generateBtn.textContent = 'Analyzing...';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred: ${error.message}`);
        } finally {
            // Hide loader and enable button
            loader.style.display = 'none';
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Content';
        }
    });

    function displayResults(data) {
        if (!data.success) {
            alert('Failed to analyze the image.');
            return;
        }

        // Populate Image and Mood
        document.getElementById('uploaded-image').src = data.image_data;
        document.getElementById('mood-text').textContent = data.mood;

        // Populate Captions
        const captionsContainer = document.getElementById('captions-container');
        captionsContainer.innerHTML = '';
        for (const [tone, caption] of Object.entries(data.captions)) {
            const card = document.createElement('div');
            card.className = 'caption-card';
            card.innerHTML = `<strong>${tone}</strong><p>${caption}</p>`;
            captionsContainer.appendChild(card);
        }

        // Populate Hashtags
        const highReachContainer = document.getElementById('high-reach-hashtags');
        highReachContainer.innerHTML = data.hashtags.high_reach.map(h => `<span class="hashtag">${h}</span>`).join('');
        
        const nicheContainer = document.getElementById('niche-hashtags');
        nicheContainer.innerHTML = data.hashtags.niche.map(h => `<span class="hashtag">${h}</span>`).join('');

        // Show the dashboard
        resultsDashboard.classList.remove('hidden');
    }
});