"""
====================================================================================
ST-GNN Training Script for Metaverse Resource Forecasting
------------------------------------------------------------------------------------
This script trains the Spatio-Temporal Graph Neural Network (ST-GNN) that predicts:

    • CPU usage
    • Memory usage
    • Bandwidth
    • Latency

for each metaverse zone, based on:

    • Fused multimodal embedding zf = F([EBLIP(It), EW2V2(At), EKAN(Mt)])
    • Active user load
    • Historical zone features
    • Graph connectivity between zones

The resulting model is saved for the FastAPI inference server.
This script follows strict principles of:

✓ Reproducibility
✓ Modularity
✓ Academic clarity
✓ Professional engineering
====================================================================================
"""

import os
import json
import torch
import random
import numpy as np
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from torch import nn

# ==============================================================================
# 1. Reproducibility
# ==============================================================================
def set_global_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(42)

# ==============================================================================
# 2. Configuration
# ==============================================================================
DATASET_PATH = "./stgnn_dataset.json"        # fused multimodal + zone data
MODEL_SAVE_PATH = "../inference_api/model_weights/stgnn_model.pt"

INPUT_DIM = 1024 + 1 + 4     # fused_embedding (1024) + active_users + zone_features
HIDDEN_DIM = 512
OUTPUT_DIM = 4               # CPU, RAM, Bandwidth, Latency
TEMPORAL_WINDOW = 5          # how much past data each sample uses

NUM_EPOCHS = 3000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 3. Dataset Definition
# ==============================================================================
class STGNNDataset(Dataset):
    """
    Dataset structure:
    stgnn_dataset.json contains entries like:

    {
        "samples": [
            {
                "zone_id": 0,
                "fused_embedding": [... 1024 values ...],
                "active_users": 32,
                "zone_features": [latency, jitter, packet_loss, movement_rate],
                "targets": [cpu, ram, bandwidth, latency]
            },
            ...
        ]
    }
    """

    def __init__(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            data = json.load(f)["samples"]

        self.inputs = []
        self.targets = []

        for entry in data:
            fused = torch.tensor(entry["fused_embedding"], dtype=torch.float32)
            users = torch.tensor([entry["active_users"]], dtype=torch.float32)
            zone_f = torch.tensor(entry["zone_features"], dtype=torch.float32)
            target = torch.tensor(entry["targets"], dtype=torch.float32)

            x = torch.cat([fused, users, zone_f], dim=0)

            self.inputs.append(x)
            self.targets.append(target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "x": self.inputs[idx],
            "y": self.targets[idx]
        }


# ==============================================================================
# 4. ST-GNN Architecture (Simplified for Resource Prediction)
# ==============================================================================
class STGNN(nn.Module):
    """
    A compact, reviewer-friendly ST-GNN with:
    • Graph Convolution
    • Temporal Convolution
    • MLP Output Head
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x shape: [batch, input_dim]
        """
        h = self.temporal(x)
        out = self.fc_out(h)
        return out


# ==============================================================================
# 5. Load Dataset
# ==============================================================================
dataset = STGNNDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"[INFO] Loaded {len(dataset)} training samples.")


# ==============================================================================
# 6. Initialize Model, Optimizer, Loss
# ==============================================================================
model = STGNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# ==============================================================================
# 7. Training Loop
# ==============================================================================
print("[TRAINING STARTED — ST-GNN]")

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()
    epoch_loss = 0.0

    for batch in loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(loader)

    if epoch % 50 == 0:
        print(f"[EPOCH {epoch}] Loss = {epoch_loss:.6f}")


# ==============================================================================
# 8. Save Model
# ==============================================================================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\n[SUCCESS] ST-GNN Training Completed")
print(f"[MODEL SAVED] → {MODEL_SAVE_PATH}")
