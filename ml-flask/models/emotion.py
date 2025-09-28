# # ml-flask/models/emotion.py

# from transformers import pipeline
# from PIL import Image
# import torch

# class EmotionAnalyzer:
#     def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
#         """
#         Initializes the emotion analysis pipeline.
#         NOTE: This is a text-based model. For a full vision solution,
#         you'd use a facial recognition model first to detect faces, then classify emotion.
#         For simplicity here, we will use a text classifier on the generated caption.
        
#         For a more advanced, vision-first approach, you could use a model like:
#         "trpakov/vit-face-expression"
#         """
#         print("Loading Emotion Analysis model...")
#         self.device = 0 if torch.cuda.is_available() else -1
#         # This pipeline is text-based and will analyze the image's caption.
#         # It's a pragmatic approach to capture the overall 'vibe' or mood.
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
            
#         results = self.classifier(text)
#         # Find the emotion with the highest score
#         dominant_emotion = max(results[0], key=lambda x: x['score'])
#         return dominant_emotion['label'].capitalize()


# # Example usage:
# if __name__ == '__main__':
#     analyzer = EmotionAnalyzer()
#     # Test with a sample caption
#     caption = "A happy dog playing in a sunny park with its owner"
#     mood = analyzer.analyze_mood_from_text(caption)
#     print(f"Caption: '{caption}'")
#     print(f"Analyzed Mood: {mood}") # Expected output: Joy
    
#     caption_2 = "A lone figure stands on a cliff overlooking a stormy sea"
#     mood_2 = analyzer.analyze_mood_from_text(caption_2)
#     print(f"Caption: '{caption_2}'")
#     print(f"Analyzed Mood: {mood_2}") # Expected output: Sadness or Fear








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