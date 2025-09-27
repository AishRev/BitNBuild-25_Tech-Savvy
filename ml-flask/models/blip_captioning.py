# ml-flask/models/blip_captioning.py

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        """
        Initializes the Image Captioning model and processor.
        Utilizes 8-bit quantization to reduce memory footprint.
        """
        print("Loading Image Captioning model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the processor and model with quantization for efficiency
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
        ).to(self.device)
        print("Image Captioning model loaded successfully.")

    def generate_caption(self, image: Image.Image) -> str:
        """
        Generates a descriptive caption for the given image.
        
        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            str: The generated caption.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        
        # Generate caption
        pixel_values = inputs.pixel_values
        
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return generated_caption

# Example usage:
if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # It's useful for testing the module in isolation
    try:
        captioner = ImageCaptioner()
        # Make sure you have an image file named 'test_image.jpg' in the same directory
        test_image = Image.open("test_image.jpg") 
        caption = captioner.generate_caption(test_image)
        print(f"Generated Caption: {caption}")
    except FileNotFoundError:
        print("Please create a 'test_image.jpg' file in the 'models' directory to test the captioner.")
    except Exception as e:
        print(f"An error occurred: {e}")