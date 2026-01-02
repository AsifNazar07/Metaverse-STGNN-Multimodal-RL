import os
import json
from PIL import Image
from torch.utils.data import Dataset


class VOC2012CaptionDataset(Dataset):

    def __init__(self, image_dir, annotation_file, processor, max_length=50):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        # Load captions
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        self.image_files = list(self.annotations.keys())
        self.captions = list(self.annotations.values())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        caption = self.captions[idx]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        enc = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        enc = {k: v.squeeze(0) for k, v in enc.items()}

        return enc
