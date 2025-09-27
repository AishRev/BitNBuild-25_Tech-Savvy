# ml-flask/models/tone_generator.py

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class ToneGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initializes the T5 model for tone generation.
        """
        print("Loading Tone Generation (T5) model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("Tone Generation model loaded successfully.")

    def generate_toned_captions(self, base_caption: str) -> dict:
        """
        Generates captions in various tones based on a base caption.

        Args:
            base_caption (str): The initial caption from the BLIP model.

        Returns:
            dict: A dictionary where keys are tones and values are the generated captions.
        """
        tones = {
            "Witty": "Rewrite this caption in a witty and humorous tone: ",
            "Inspirational": "Rewrite this caption in an inspirational and uplifting tone: ",
            "Professional": "Rewrite this caption in a formal and professional tone for a business audience: ",
            "Casual": "Rewrite this caption in a casual and friendly tone, like for a social media post: "
        }
        
        toned_captions = {}
        for tone, prompt in tones.items():
            input_text = prompt + base_caption
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            outputs = self.model.generate(input_ids, max_length=60, num_beams=4, early_stopping=True)
            toned_caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            toned_captions[tone] = toned_caption
            
        return toned_captions

# Example usage:
if __name__ == '__main__':
    generator = ToneGenerator()
    base_caption = "a group of friends laughing together at a beach bonfire"
    captions = generator.generate_toned_captions(base_caption)
    
    print(f"Base Caption: {base_caption}\n")
    for tone, caption in captions.items():
        print(f"[{tone}]: {caption}")