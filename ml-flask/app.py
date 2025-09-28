import os
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
import traceback

# Import our custom model classes
from models.blip_captioning import ImageCaptioner
from models.detectors import ObjectDetector
from models.emotion import EmotionAnalyzer # Using the new vision-based version
from models.clip_utils import ThemeExtractor
from models.tone_generator import ToneGenerator

# --- App Initialization ---
app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Pre-load ML Models ---
print("--- Initializing AI Models ---")
captioner = ImageCaptioner()
detector = ObjectDetector()
mood_analyzer = EmotionAnalyzer()
theme_extractor = ThemeExtractor()
tone_generator = ToneGenerator()
print("--- All Models Initialized Successfully ---")

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_hashtags(objects: list, themes: list) -> dict:
    """
    Generates categorized hashtags from detected objects and themes.
    """
    high_reach = [f"#{theme.replace(' ', '')}" for theme in themes]
    high_reach.extend(['#AIContent', '#PhotoOfTheDay', '#InstaGood'])
    
    niche = [f"#{obj.replace(' ', '')}" for obj in objects]
    
    return {
        "high_reach": list(set(high_reach))[:5],
        "niche": list(set(niche))[:10]
    }

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    The core API endpoint with the corrected pipeline logic.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img = Image.open(filepath)
            
            # --- CORRECTED AI PIPELINE EXECUTION ---
            
            # 1. Generate Base Caption
            base_caption = captioner.generate_caption(img)
            
            # 2. Generate Toned Captions
            toned_captions = tone_generator.generate_toned_captions(base_caption)
            toned_captions['Original'] = base_caption
            
            # 3. Analyze Mood directly from the IMAGE
            # THIS IS THE CORRECTED LINE
            mood = mood_analyzer.analyze_mood_from_image(img)
            
            # 4. Detect Objects
            objects = detector.detect_objects(img)
            
            # 5. Extract Themes
            candidate_themes = [
                'Nature', 'Urban', 'Technology', 'Food', 'Portrait', 'Travel', 
                'Art', 'Fashion', 'Sports', 'Animals', 'Business', 'Abstract'
            ]
            themes = theme_extractor.extract_themes(img, candidate_themes)
            
            # 6. Generate Hashtags
            hashtags = generate_hashtags(objects, themes)

            # --- Prepare Response ---
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            response_data = {
                "success": True,
                "image_data": f"data:image/jpeg;base64,{img_str}",
                "captions": toned_captions,
                "mood": mood,
                "hashtags": hashtags
            }
            
            return jsonify(response_data)

        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            traceback.print_exc()
            return jsonify({"error": "Failed to process image", "details": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

