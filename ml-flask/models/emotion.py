# ml-flask/models/emotion.py

from transformers import pipeline
from PIL import Image
import torch

class EmotionAnalyzer:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        """
        Initializes the emotion analysis pipeline.
        NOTE: This is a text-based model. For a full vision solution,
        you'd use a facial recognition model first to detect faces, then classify emotion.
        For simplicity here, we will use a text classifier on the generated caption.
        
        For a more advanced, vision-first approach, you could use a model like:
        "trpakov/vit-face-expression"
        """
        print("Loading Emotion Analysis model...")
        self.device = 0 if torch.cuda.is_available() else -1
        # This pipeline is text-based and will analyze the image's caption.
        # It's a pragmatic approach to capture the overall 'vibe' or mood.
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=self.device
        )
        print("Emotion Analysis model loaded successfully.")

    def analyze_mood_from_text(self, text: str) -> str:
        """
        Analyzes the emotion/mood from a given text string (like an image caption).

        Args:
            text (str): The input text.

        Returns:
            str: The dominant emotion (e.g., 'joy', 'sadness').
        """
        if not text:
            return "Neutral"
            
        results = self.classifier(text)
        # Find the emotion with the highest score
        dominant_emotion = max(results[0], key=lambda x: x['score'])
        return dominant_emotion['label'].capitalize()


# Example usage:
if __name__ == '__main__':
    analyzer = EmotionAnalyzer()
    # Test with a sample caption
    caption = "A happy dog playing in a sunny park with its owner"
    mood = analyzer.analyze_mood_from_text(caption)
    print(f"Caption: '{caption}'")
    print(f"Analyzed Mood: {mood}") # Expected output: Joy
    
    caption_2 = "A lone figure stands on a cliff overlooking a stormy sea"
    mood_2 = analyzer.analyze_mood_from_text(caption_2)
    print(f"Caption: '{caption_2}'")
    print(f"Analyzed Mood: {mood_2}") # Expected output: Sadness or Fear