"""
====================================================================================
BLIP Fine-Tuning Pipeline on PASCAL VOC 2012
------------------------------------------------------------------------------------
This script implements a fully reproducible fine-tuning pipeline for the BLIP
image–text encoder EBLIP(It), aligned with the methodology described in our paper.

• Model: Salesforce BLIP (Image Captioning Base)
• Dataset: PASCAL VOC 2012 (Image–Caption pairs)
• Epochs: 20,000 (long-horizon convergence for low batch-size training)
• Batch Size: 1 (stable gradient updates under heavy GPU memory constraints)

====================================================================================
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from PIL import Image
from typing import Dict, Any


# ==============================================================================
# 1. Reproducibility Configuration
# ==============================================================================
def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(42)


# ==============================================================================
# 2. Configuration Constants (EDIT PATHS BEFORE TRAINING)
# ==============================================================================
IMAGE_DIR = "PATH/TO/VOC2012/JPEGImages"
ANNOTATION_FILE = "PATH/TO/VOC2012/Annotations.json"

OUTPUT_DIR = "./EBLIP_VOC2012_Finetuned"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

NUM_EPOCHS = 20000     
TRAIN_BATCH_SIZE = 1    # Ultra-stable small batch for fine-grained gradients
EVAL_BATCH_SIZE = 1


# ==============================================================================
# 3. Load Processor & Model
# ==============================================================================
print("\n[INFO] Loading BLIP model and processor...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ==============================================================================
# 4. Dataset Definition (VOC2012 Image–Caption Loader)
# ==============================================================================
class VOC2012Dataset(Dataset):
    """
    VOC2012 dataset loader for image–caption fine-tuning.

    Expected annotation JSON structure:
    {
        "2008_000008.jpg": "A dog jumping over a pole...",
        "2008_000024.jpg": "Two cars parked on the street...",
        ...
    }
    """

    def __init__(self, image_dir: str, annotation_file: str, processor: BlipProcessor):
        self.image_dir = image_dir
        self.processor = processor

        with open(annotation_file, "r") as f:
            self.annotations: Dict[str, str] = json.load(f)

        self.image_files = list(self.annotations.keys())
        self.captions = list(self.annotations.values())

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, filename: str) -> Image.Image:
        path = os.path.join(self.image_dir, filename)
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.image_files[idx]
        caption = self.captions[idx]

        image = self._load_image(img_name)

        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
            truncation=True
        )

        labels = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
            truncation=True
        )["input_ids"]

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels.squeeze(0)
        inputs["labels"] = labels

        return inputs


# ==============================================================================
# 5. Dataset Loading & Splitting
# ==============================================================================
print("\n[INFO] Initializing VOC2012 dataset...")
dataset = VOC2012Dataset(IMAGE_DIR, ANNOTATION_FILE, processor)

train_size = int(0.85 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

print(f"[INFO] Dataset size: {len(dataset)} samples")
print(f"[INFO] Train / Eval split: {train_size} / {eval_size}")


# ==============================================================================
# 6. Data Collator for Sequence Padding
# ==============================================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    padding=True
)


# ==============================================================================
# 7. TrainingArguments Setup
# ==============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=1,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=1000,
)

print("\n[INFO] Training configuration loaded.")


# ==============================================================================
# 8. Trainer API
# ==============================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)


# ==============================================================================
# 9. Training Execution
# ==============================================================================
print("\n[TRAINING STARTED]")
trainer.train()


# ==============================================================================
# 10. Save Final Model
# ==============================================================================
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\n[SUCCESS] BLIP Fine-Tuning Completed")
print(f"[MODEL SAVED] → {OUTPUT_DIR}")
