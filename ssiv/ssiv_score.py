"""
SSIV Score Computation
Combines semantic and structural validation
to produce final SSIV confidence score S_t.
"""
from typing import Dict, Any
import networkx as nx

from .semantic_validator import semantic_similarity
from .structural_validator import structural_consistency
from .ssiv_config import LAMBDA_SEM, LAMBDA_STR, TAU_THRESHOLD


def compute_ssiv_score(
    intent_text: str,
    policy_text: str,
    policy_graph: nx.Graph,
    forecast_graph: nx.Graph,
) -> Dict[str, Any]:
    """
    Compute SSIV confidence score S_t and validation decision.

    Args:
        intent_text (str): Multimodal-generated intent text
        policy_text (str): IBN-translated policy description
        policy_graph (nx.Graph): Policy dependency graph
        forecast_graph (nx.Graph): ST-GNN forecast-aware graph

    Returns:
        Dict with:
            - S_t (float): final SSIV score
            - semantic_score (float)
            - structural_score (float)
            - accepted (bool)
    """

    # --- Semantic validation ---
    semantic_score = semantic_similarity(
        intent_text=intent_text,
        policy_text=policy_text
    )

    # --- Structural validation ---
    structural_score = structural_consistency(
        policy_graph=policy_graph,
        forecast_graph=forecast_graph
    )

    # --- Hybrid SSIV score (Eq. 2) ---
    S_t = (
        LAMBDA_SEM * semantic_score
        + LAMBDA_STR * structural_score
    )

    accepted = S_t >= TAU_THRESHOLD

    return {
        "S_t": float(S_t),
        "semantic_score": float(semantic_score),
        "structural_score": float(structural_score),
        "accepted": accepted,
    }
