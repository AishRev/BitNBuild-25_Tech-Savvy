# import torch
# from PIL import Image
# from transformers import BlipForConditionalGeneration, BlipProcessor, T5ForConditionalGeneration, T5Tokenizer
# import requests

# class AdvancedCaptioner:
#     def __init__(self):
#         """
#         Initializes a hybrid model system for efficient captioning.
#         - Uses Salesforce/blip-large for image-to-text generation.
#         - Uses google/flan-t5-base for text-to-text tone generation.
#         This approach significantly reduces memory and disk space requirements.
#         """
#         print("Loading Compact Captioning Models (BLIP + T5)...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {self.device}")

#         # --- Load Image Captioning Model (BLIP) ---
#         self.caption_model_name = "Salesforce/blip-image-captioning-large"
#         self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
#         self.caption_model = BlipForConditionalGeneration.from_pretrained(
#             self.caption_model_name, torch_dtype=torch.float16
#         ).to(self.device)

#         # --- Load Tone Generation Model (Flan-T5) ---
#         self.tone_model_name = "google/flan-t5-base"
#         self.tone_tokenizer = T5Tokenizer.from_pretrained(self.tone_model_name)
#         self.tone_model = T5ForConditionalGeneration.from_pretrained(
#             self.tone_model_name
#         ).to(self.device)
        
#         print("Compact models loaded successfully.")

#     def generate_descriptive_analysis(self, image: Image.Image) -> dict:
#         """
#         Generates a factual base description with BLIP and then crafts expressive,
#         platform-specific social media captions with T5.
#         """
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         # --- Step 1: Generate a more detailed factual description with BLIP ---
#         inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device, torch.float16)
#         pixel_values = inputs.pixel_values
        
#         # Increased max_length to get more details for the prompts
#         generated_ids = self.caption_model.generate(pixel_values=pixel_values, max_length=100)
#         base_description = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

#         # --- Step 2: Generate Expressive, Toned Captions with new prompts ---
#         # Renamed for clarity
#         toned_captions = {"Factual Description": base_description}
        
#         # --- NEW ADVANCED PROMPTS ---
#         # These prompts are designed to be more creative and less descriptive.
#         tones = {
#             "Instagram Post": f"Act as a social media expert. The image shows '{base_description}'. Write three creative and short Instagram caption options. Focus on themes of friendship, fun, and making memories. Include relevant emojis and an engaging question.",
#             "LinkedIn Post": f"Act as a business professional. The image shows '{base_description}'. Write a short, insightful LinkedIn post about the importance of team bonding and camaraderie outside of work. End with professional hashtags like #TeamBuilding #WorkLifeBalance.",
#             "Witty One-Liner": f"An image is described as: '{base_description}'. Generate a funny or witty one-liner caption for it.",
#             "Inspirational Quote": f"The scene is '{base_description}'. Write a short, powerful, and inspirational quote about the moment. Focus on themes like unity, happiness, or cherishing time together."
#         }

#         for tone, prompt_text in tones.items():
#             input_ids = self.tone_tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
#             # Increased max_length to allow for longer, more creative captions
#             outputs = self.tone_model.generate(input_ids, max_length=120, num_beams=5, early_stopping=True)
#             toned_caption = self.tone_tokenizer.decode(outputs[0], skip_special_tokens=True)
#             toned_captions[tone] = toned_caption
            
#         return toned_captions

# # Example Usage
# if __name__ == '__main__':
#     try:
#         captioner = AdvancedCaptioner()
#         # A photo of friends on a mountain, perfect for testing the new prompts
#         url = "https://images.unsplash.com/photo-1534947921876-85b86c476a0e"
#         image = Image.open(requests.get(url, stream=True).raw)
        
#         analysis = captioner.generate_descriptive_analysis(image)
#         print("\n--- Generated Content (New Prompts) ---")
#         for tone, caption in analysis.items():
#             print(f"[{tone}]: {caption}\n")
            
#     except Exception as e:
#         print(f"An error occurred: {e}")


