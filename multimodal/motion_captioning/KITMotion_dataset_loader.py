
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class KITMotionDataset(Dataset):
    """
    Loads motion sequences + language captions for KAN motion encoder training.
    """

    def __init__(self, motion_dir, text_dir, tokenizer=None, max_frames=200):
        self.motion_dir = motion_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.max_frames = max_frames

        self.motion_files = sorted([
            f for f in os.listdir(motion_dir) if f.endswith(".npy")
        ])

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.motion_files)

    # ----------------------------------------------------------------------
    def load_motion(self, motion_path):
        """
        Loads a motion sequence and trims/pads to max_frames.
        """
        motion = np.load(motion_path)

        # Trim long sequences
        if motion.shape[0] > self.max_frames:
            motion = motion[:self.max_frames]

        # Pad short sequences
        if motion.shape[0] < self.max_frames:
            pad_len = self.max_frames - motion.shape[0]
            pad_block = np.zeros((pad_len, motion.shape[1]))
            motion = np.vstack([motion, pad_block])

        return torch.tensor(motion, dtype=torch.float32)

    # ----------------------------------------------------------------------
    def load_text(self, text_path):
        """
        Loads text description and tokenizes if tokenizer is provided.
        """
        with open(text_path, "r") as f:
            caption = f.read().strip()

        if self.tokenizer:
            enc = self.tokenizer(
                caption,
                return_tensors="pt",
                max_length=60,
                truncation=True,
                padding="max_length"
            )
            return enc["input_ids"].squeeze(0)

        return caption

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        motion_file = self.motion_files[idx]
        motion_path = os.path.join(self.motion_dir, motion_file)

        text_file = motion_file.replace(".npy", ".txt")
        text_path = os.path.join(self.text_dir, text_file)

        motion_tensor = self.load_motion(motion_path)
        text_encoded = self.load_text(text_path)

        return {
            "motion": motion_tensor,
            "caption": text_encoded
        }
