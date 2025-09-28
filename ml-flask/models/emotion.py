from transformers import pipeline
from PIL import Image
import torch
import cv2
import numpy as np

class EmotionAnalyzer:
    def __init__(self, model_name="trpakov/vit-face-expression"):
        """
        Initializes a VISION-BASED pipeline for facial emotion recognition.
        This model first detects faces in the image and then classifies the
        expression on each face, making it much more accurate than text analysis.
        This approach also avoids the recent PyTorch security issue.
        """
        print("Loading VISION-BASED Emotion Analysis model...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "image-classification",
            model=model_name,
            device=self.device
        )
        # Load a pre-trained face detector from OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Vision-Based Emotion Analysis model loaded successfully.")

    def analyze_mood_from_image(self, image: Image.Image) -> str:
        """
        Analyzes the dominant emotion from faces detected directly in an image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            str: The dominant emotion found among the detected faces (e.g., 'Happy', 'Sad').
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert PIL Image to an OpenCV image format
        open_cv_image = np.array(image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return "Neutral" # Return "Neutral" if no faces are found

        # Tally the emotions found on each face
        emotions = {}
        for (x, y, w, h) in faces:
            # Extract the face from the image
            face_roi = image.crop((x, y, x + w, y + h))
            
            # Classify the emotion of the face
            results = self.classifier(face_roi)
            dominant_emotion = results[0]['label'].capitalize()
            
            emotions[dominant_emotion] = emotions.get(dominant_emotion, 0) + 1
        
        # Return the most frequently detected emotion
        if not emotions:
            return "Neutral"
        
        dominant_mood = max(emotions, key=emotions.get)
        return dominant_mood




# from transformers import pipeline
# from PIL import Image
# import torch
# import cv2
# import numpy as np

# class EmotionAnalyzer:
#     def __init__(self, model_name="michellejieli/emotion_text_classifier"):
#         """
#         Initializes the emotion analysis pipeline.
#         This version uses a modern, safetensors-based model that is compliant
#         with the latest security updates in the transformers library.
        
#         NOTE: This is a text-based model. For the vision-based approach,
#         please see the alternative code in the comments.
#         """
#         print("Loading Emotion Analysis model...")
#         self.device = 0 if torch.cuda.is_available() else -1
#         # This pipeline is text-based and will analyze the image's caption.
#         self.classifier = pipeline(
#             "text-classification",
#             model=model_name,
#             return_all_scores=True,
#             device=self.device
#         )
#         print("Emotion Analysis model loaded successfully.")

#     def analyze_mood_from_text(self, text: str) -> str:
#         """
#         Analyzes the emotion/mood from a given text string (like an image caption).

#         Args:
#             text (str): The input text.

#         Returns:
#             str: The dominant emotion (e.g., 'joy', 'sadness').
#         """
#         if not text:
#             return "Neutral"
            
#         try:
#             results = self.classifier(text)
#             # The output format can vary, so we handle both list-of-lists and list-of-dicts
#             if isinstance(results[0], list):
#                 scores = results[0]
#             else:
#                 scores = results

#             dominant_emotion = max(scores, key=lambda x: x['score'])
#             return dominant_emotion['label'].capitalize()
#         except Exception as e:
#             print(f"Could not analyze emotion from text: {e}")
#             return "Neutral"

# # Example usage:
# if __name__ == '__main__':
#     analyzer = EmotionAnalyzer()
#     caption = "A happy dog playing in a sunny park with its owner"
#     mood = analyzer.analyze_mood_from_text(caption)
#     print(f"Caption: '{caption}'")
#     print(f"Analyzed Mood: {mood}") 

