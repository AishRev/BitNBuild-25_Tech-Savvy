# ml-flask/scripts/finetune_blip.py

# NOTE: This is a complex and resource-intensive task. 
# Running this requires a powerful GPU and a significant amount of time.
# This script serves as a template and starting point.

import torch
from datasets import load_dataset
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from PIL import Image
import requests

# --- 1. Configuration ---
MODEL_NAME = "Salesforce/blip-image-captioning-base" # Use base model for faster fine-tuning
DATASET_NAME = "conceptual_captions" # Example dataset from Hugging Face
OUTPUT_DIR = "./blip-finetuned-conceptual-captions"

# --- 2. Load Processor and Model ---
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- 3. Load and Preprocess Dataset ---
# We'll only use a small subset for this example to make it runnable.
# For real fine-tuning, you would use the full dataset.
dataset = load_dataset(DATASET_NAME, split="train[:1000]")

def preprocess_data(examples):
    """
    Preprocesses a batch of examples from the dataset.
    Downloads images and prepares them for the model.
    """
    images = []
    # Some URLs might be broken, so we wrap in a try-except block
    for image_url in examples['image_url']:
        try:
            image = Image.open(requests.get(image_url, stream=True, timeout=5).raw).convert("RGB")
            images.append(image)
        except Exception:
            # Append a blank image if one fails to download
            images.append(Image.new('RGB', (224, 224)))
            
    captions = examples['caption']
    
    # Process images and captions
    inputs = processor(images=images, text=captions, padding="max_length", return_tensors="pt")
    
    # The model expects 'labels' for the text part during training
    inputs['labels'] = inputs['input_ids']
    
    return inputs

# Apply preprocessing. This can take a while.
processed_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
processed_dataset.set_format("torch")

# --- 4. Define Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                     # Number of training epochs
    per_device_train_batch_size=8,          # Batch size per device during training
    save_steps=500,                         # Save checkpoint every 500 steps
    save_total_limit=2,                     # Only keep the last 2 checkpoints
    logging_steps=50,                       # Log metrics every 50 steps
    fp16=True,                              # Use mixed precision for faster training on compatible GPUs
    push_to_hub=False,                      # Set to True if you want to upload to Hugging Face Hub
)

# --- 5. Initialize the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    # You can also provide a 'compute_metrics' function for evaluation
)

# --- 6. Start Training ---
print("Starting fine-tuning process...")
trainer.train()
print("Fine-tuning complete.")

# --- 7. Save the final model ---
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
processor.save_pretrained(f"{OUTPUT_DIR}/final_model")
print(f"Model saved to {OUTPUT_DIR}/final_model")

# To use your fine-tuned model later:
# processor = BlipProcessor.from_pretrained("./blip-finetuned-conceptual-captions/final_model")
# model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned-conceptual-captions/final_model")