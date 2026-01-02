
import os
import torch
import random
import numpy as np
from torch import nn, optim
from fusion_model import FusionModel
from metrics.fusion_metrics import contrastive_loss, triplet_loss


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------
class FusionDataset(torch.utils.data.Dataset):


    def __init__(self, img_arr, audio_arr, motion_arr, labels):
        self.img_arr = img_arr
        self.audio_arr = audio_arr
        self.motion_arr = motion_arr
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.img_arr[idx], dtype=torch.float32),
            torch.tensor(self.audio_arr[idx], dtype=torch.float32),
            torch.tensor(self.motion_arr[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------
def train_fusion_model(
    img_embeddings,
    audio_embeddings,
    motion_embeddings,
    labels,
    save_path="fusion_model.pth",
    epochs=20,
    batch_size=32,
    lr=1e-4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FusionDataset(img_embeddings, audio_embeddings, motion_embeddings, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Embedding dimension detection
    d_img = img_embeddings.shape[1]
    d_audio = audio_embeddings.shape[1]
    d_motion = motion_embeddings.shape[1]

    # Create fusion model
    model = FusionModel(d_img, d_audio, d_motion, fused_dim=1024)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n[INFO] Training Fusion Model (EFUS)")
    print("------------------------------------------------------------")
    print(f"Image embedding dim   : {d_img}")
    print(f"Audio embedding dim   : {d_audio}")
    print(f"Motion embedding dim  : {d_motion}")
    print(f"Fused embedding dim   : 1024")
    print(f"Dataset size          : {len(dataset)} samples\n")

    for epoch in range(1, epochs + 1):

        total_loss = 0

        for img, aud, mot, label in loader:

            img = img.to(device)
            aud = aud.to(device)
            mot = mot.to(device)
            label = label.to(device)

            fused = model(img, aud, mot)

            # Contrastive or triplet loss depending on label structure
            if label.ndim == 1:  # binary labels
                loss = contrastive_loss(fused, label)
            else:                # triplet structure (anchor, positive, negative)
                loss = triplet_loss(*fused)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[EPOCH {epoch:03d}] Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n[SAVED] Fusion model → {save_path}")


# ------------------------------------------------------------------------------
# Main Execution for Quick Testing
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Fake sample data for quick execution (testing purposes)
    num_samples = 100
    d_img, d_audio, d_motion = 768, 768, 512

    img_embeds = np.random.randn(num_samples, d_img)
    aud_embeds = np.random.randn(num_samples, d_audio)
    motion_embeds = np.random.randn(num_samples, d_motion)
    labels = np.random.randint(0, 2, size=num_samples)

    train_fusion_model(
        img_embeddings=img_embeds,
        audio_embeddings=aud_embeds,
        motion_embeddings=motion_embeds,
        labels=labels,
        epochs=5,
        save_path="fusion_dummy.pth"
    )
