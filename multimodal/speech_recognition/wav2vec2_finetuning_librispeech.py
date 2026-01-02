
#Wav2Vec2 Fine-Tuning Pipeline on LibriSpeech Dataset)


import os
import torch
import random
import numpy as np
import torchaudio
from torch.utils.data import Dataset, random_split
from typing import Dict, Any
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)

# ==============================================================================
# 1. Reproducibility Configuration
# ==============================================================================
def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(42)

# ==============================================================================
# 2. Experiment Configuration (EDIT PATHS BEFORE TRAINING)
# ==============================================================================
LIBRISPEECH_DIR = "PATH/TO/LibriSpeech/train-clean-100"
OUTPUT_DIR = "./EW2V2_LibriSpeech_Finetuned"

# Load model from local directory
MODEL_NAME = "./my_wav2vec2_model/"

TARGET_SR = 16000
NUM_EPOCHS = 20000
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

# ==============================================================================
# 3. Load Processor + Model from Local Directory
# ==============================================================================
print("\n[INFO] Loading local Wav2Vec2 model from disk...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==============================================================================
# 4. LibriSpeech Dataset Loader
# ==============================================================================
class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset loader for ASR fine-tuning.
    Each sample consists of an audio waveform and its corresponding transcription.
    Inputs are processed using Wav2Vec2Processor for model compatibility.
    """

    def __init__(self, data_dir: str, processor: Wav2Vec2Processor):
        self.processor = processor
        self.data_dir = data_dir
        self.files = []
        self.transcripts = {}

        print("[INFO] Parsing LibriSpeech directory...\n")

        # Load transcription text
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".trans.txt"):
                    trans_path = os.path.join(root, f)
                    with open(trans_path, "r") as file:
                        for line in file.readlines():
                            parts = line.strip().split(" ")
                            uid = parts[0]
                            text = " ".join(parts[1:]).upper()
                            self.transcripts[uid] = text

        # Load audio paths
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".flac"):
                    uid = f.replace(".flac", "")
                    if uid in self.transcripts:
                        self.files.append(os.path.join(root, f))

        print(f"[INFO] Loaded {len(self.files)} audio-text pairs.\n")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audio_path = self.files[idx]
        uid = os.path.basename(audio_path).replace(".flac", "")
        text = self.transcripts[uid]

        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

        waveform = waveform.squeeze(0)

        # Process audio input
        inputs = processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="longest"
        )

        # Process label text
        with processor.as_target_processor():
            labels = processor(text, return_tensors="pt").input_ids

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": labels.squeeze(0)
        }

# ==============================================================================
# 5. Load Dataset + Train/Eval Split
# ==============================================================================
print("[INFO] Initializing LibriSpeech dataset...")
dataset = LibriSpeechDataset(LIBRISPEECH_DIR, processor)

train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

print(f"[INFO] Dataset size: {len(dataset)} samples")
print(f"[INFO] Train/Eval split: {train_size}/{eval_size}\n")
# ==============================================================================
# 6. Data Collator for Variable-Length Sequences
# ==============================================================================
def collate_fn(batch):
    """Handle variable-length sequences for ASR training."""
    audio = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch_inputs = processor.pad(
        {"input_values": audio},
        return_tensors="pt"
    )

    with processor.as_target_processor():
        batch_labels = processor.pad(
            {"input_ids": labels},
            return_tensors="pt"
        )

    batch_inputs["labels"] = batch_labels["input_ids"]
    return batch_inputs

# ==============================================================================
# 7. TrainingArguments Setup
# ==============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=1e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="none",
)

print("[INFO] TrainingArguments initialized.\n")

# ==============================================================================
# 8. Trainer API
# ==============================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

# ==============================================================================
# 9. Start Training
# ==============================================================================
print("[TRAINING STARTED Fine-Tuning]\n")
trainer.train()

# ==============================================================================
# 10. Save Final Model
# ==============================================================================
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\n[SUCCESS] Wav2Vec2 Fine-Tuning Completed")
print(f"[MODEL SAVED] → {OUTPUT_DIR}")
