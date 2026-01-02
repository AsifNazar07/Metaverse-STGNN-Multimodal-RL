"""
====================================================================================
ST-GNN Inference Module
------------------------------------------------------------------------------------
This module loads the trained Spatio-Temporal Graph Neural Network (ST-GNN)
and exposes a simple Python API for performing inference.

The FastAPI server (main.py) will import this module to execute:

    prediction = stgnn_predict(input_features)

Input format from API:
{
    "fused_embedding": [... 1024 dims ...],
    "active_users": 32,
    "zone_features": [latency, jitter, packet_loss, movement_rate]
}

Output format:
{
    "cpu": float,
    "memory": float,
    "bandwidth": float,
    "latency": float
}

====================================================================================
"""

import os
import torch
import numpy as np
from torch import nn

# ==============================================================================
# 1. Model Configuration
# ==============================================================================
INPUT_DIM = 1024 + 1 + 4      # fused_embedding + active_users + zone_features
HIDDEN_DIM = 512
OUTPUT_DIM = 4                # cpu, memory, bandwidth, latency

MODEL_WEIGHTS_PATH = "./model_weights/stgnn_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# 2. ST-GNN Architecture Definition
# ==============================================================================
class STGNN(nn.Module):
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
        return self.fc_out(self.temporal(x))


# ==============================================================================
# 3. Model Loader
# ==============================================================================
def load_stgnn_model():
    model = STGNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] ST-GNN model loaded from: {MODEL_WEIGHTS_PATH}")
    return model


model = load_stgnn_model()


# ==============================================================================
# 4. Inference Function
# ==============================================================================
def stgnn_predict(fused_embedding, active_users, zone_features):
    """
    Runs a single prediction step on the ST-GNN model.

    Args:
        fused_embedding (list[float]): 1024-D fused multimodal embedding
        active_users (int/float): number of active users
        zone_features (list[float]): [latency, jitter, packet_loss, movement]

    Returns:
        dict: predicted resource usage
    """

    # Convert to tensor
    fused_embedding = torch.tensor(fused_embedding, dtype=torch.float32)
    active_users = torch.tensor([active_users], dtype=torch.float32)
    zone_features = torch.tensor(zone_features, dtype=torch.float32)

    # Concatenate inputs into single feature vector
    x = torch.cat([fused_embedding, active_users, zone_features], dim=0)
    x = x.to(DEVICE)

    # Forward pass
    with torch.no_grad():
        pred = model(x)

    cpu, memory, bandwidth, latency = pred.cpu().numpy()

    return {
        "cpu": float(cpu),
        "memory": float(memory),
        "bandwidth": float(bandwidth),
        "latency": float(latency)
    }
