# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModel
# import requests

# class AestheticScorer:
#     def __init__(self, model_name="shunk031/aesthetics-predictor-v2-vit-large-patch14-224"):
#         """
#         Initializes a model that predicts the aesthetic score of an image.
#         This version uses a stable, publicly available model.
#         """
#         print("Loading Aesthetic Scorer model...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # This will download the model from the new, stable location
#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         print("Aesthetic Scorer model loaded successfully.")

#     def score_image(self, image: Image.Image) -> dict:
#         """
#         Scores an image on its aesthetic quality.

#         Args:
#             image (PIL.Image.Image): The input image.

#         Returns:
#             dict: A dictionary containing the score and a qualitative judgment.
#         """
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         inputs = self.processor(images=image, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         # The model outputs a score, typically between 1 and 10
#         aesthetic_score = outputs.logits[0].item()
        
#         # Round the score to two decimal places
#         score = round(aesthetic_score, 2)
        
#         # Provide a human-readable judgment based on the score
#         if score > 7.5:
#             judgment = "Excellent! Great lighting and composition."
#         elif score > 6.0:
#             judgment = "Very Good! A well-composed and appealing shot."
#         elif score > 4.5:
#             judgment = "Good. A solid photo with clear potential."
#         else:
#             judgment = "Standard. Could be improved with better lighting or focus."
            
#         return {"score": score, "judgment": judgment}

# # Example Usage to test this file directly
# if __name__ == '__main__':
#     try:
#         scorer = AestheticScorer()
#         url = "https://images.unsplash.com/photo-1575936123452-b67c3203c357"
#         image = Image.open(requests.get(url, stream=True).raw)
        
#         result = scorer.score_image(image)
#         print(f"Aesthetic Score: {result['score']}/10")
#         print(f"Judgment: {result['judgment']}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

