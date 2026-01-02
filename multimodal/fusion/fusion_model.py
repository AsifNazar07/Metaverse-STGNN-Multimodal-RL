"""
====================================================================================
Multimodal Fusion Model for Image–Audio–Motion Embeddings
====================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple


# ==============================================================================
# 1. Fusion Configuration (EDIT IF YOU CHANGE EMBEDDING DIMENSIONS)
# ==============================================================================
IMG_DIM = 512    # Output dim of BLIP 
AUD_DIM = 768    # Output dim of Wav2Vec2 
MOT_DIM = 512    # Output dim of KAN
FUSED_DIM = 1024 # Dimensionality of the final fused vector zf


# ==============================================================================
# 2. Fusion Network Architecture
# ==============================================================================
class MultimodalFusion(nn.Module):
    """
    Multimodal Fusion Network
    Combines (img_emb, aud_emb, mot_emb) into a shared representation zf.

    Architecture:
        • Concatenation of embeddings
        • Two-layer MLP with normalization
        • Nonlinear activation
    """

    def __init__(
        self,
        img_dim: int = IMG_DIM,
        aud_dim: int = AUD_DIM,
        mot_dim: int = MOT_DIM,
        fused_dim: int = FUSED_DIM
    ):
        super().__init__()

        input_dim = img_dim + aud_dim + mot_dim

        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),

            nn.Linear(1024, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim)
        )

    def forward(
        self,
        img_emb: torch.Tensor,
        aud_emb: torch.Tensor,
        mot_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for fusion.

        Args:
            img_emb (Tensor): Image embedding dv
            aud_emb (Tensor): Audio embedding da
            mot_emb (Tensor): Motion embedding dm

        Returns:
            fused embedding zf ∈ R^FUSED_DIM
        """

        x = torch.cat([img_emb, aud_emb, mot_emb], dim=-1)
        zf = self.fusion_net(x)
        return zf


# ==============================================================================
# 3. Fusion Inference Wrapper
# ==============================================================================
def fuse_embeddings(
    img_emb: torch.Tensor,
    aud_emb: torch.Tensor,
    mot_emb: torch.Tensor,
    fusion_model: MultimodalFusion
) -> torch.Tensor:
    """
    Convenience wrapper for fusion inference.

    Example:
        zf = fuse_embeddings(img_vec, aud_vec, mot_vec, fusion_model)
    """
    return fusion_model(img_emb, aud_emb, mot_emb)


# ==============================================================================
# 4. Example Usage
# ==============================================================================
if __name__ == "__main__":
    print("[INFO] Running a test forward pass for fusion...")

    fusion_model = MultimodalFusion()

    img_dummy = torch.randn(1, IMG_DIM)
    aud_dummy = torch.randn(1, AUD_DIM)
    mot_dummy = torch.randn(1, MOT_DIM)

    zf = fusion_model(img_dummy, aud_dummy, mot_dummy)
    print(f"[INFO] Fused embedding shape: {zf.shape}")
    print(f"[INFO] Example forward pass successful.")
