# ml-flask/models/clip_utils.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ThemeExtractor:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        """
        Initializes the CLIP model for theme extraction.
        """
        print("Loading Theme Extraction (CLIP) model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("Theme Extraction (CLIP) model loaded successfully.")

    def extract_themes(self, image: Image.Image, candidate_themes: list) -> list:
        """
        Identifies the most relevant themes for an image from a list of candidates.

        Args:
            image (PIL.Image.Image): The input image.
            candidate_themes (list): A list of strings representing potential themes.

        Returns:
            list: A list of the top 3 most relevant themes.
        """
        inputs = self.processor(
            text=candidate_themes, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # This gives the similarity score between the image and each text theme
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Pair themes with their probabilities and sort
        theme_probs = sorted(zip(candidate_themes, probs), key=lambda x: x[1], reverse=True)
        
        # Return the top 3 themes
        top_themes = [theme for theme, prob in theme_probs[:3]]
        return top_themes

# Example usage:
if __name__ == '__main__':
    try:
        extractor = ThemeExtractor()
        test_image = Image.open("test_image.jpg")
        themes = [
            'Nature', 'Cityscape', 'Indoors', 'Food', 'Portrait', 'Technology', 
            'Travel', 'Art', 'Fashion', 'Sports', 'Animals'
        ]
        top_themes = extractor.extract_themes(test_image, themes)
        print(f"Top Themes: {top_themes}")
    except FileNotFoundError:
        print("Please create a 'test_image.jpg' file in the 'models' directory to test the theme extractor.")
    except Exception as e:
        print(f"An error occurred: {e}")