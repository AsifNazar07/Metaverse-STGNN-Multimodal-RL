"""
SSIV Configuration Parameters

Defines thresholds and weighting factors
used in semantic–structural intent verification.
"""

# -----------------------------
# SSIV acceptance threshold τ
# -----------------------------
# Intents with S_t < τ are rejected or re-generated
TAU_THRESHOLD: float = 0.8


# ---------------------------------------
# Weighting factors for hybrid SSIV score
# ---------------------------------------
# λ1 + λ2 = 1.0 (recommended for stability)

LAMBDA_SEM: float = 0.5   # Semantic similarity weight (Sentence-BERT)
LAMBDA_STR: float = 0.5   # Structural consistency weight (GAT)


# -----------------------------
# Sanity checks 
# -----------------------------
assert 0.0 <= TAU_THRESHOLD <= 1.0, "τ must be in [0, 1]"
assert abs((LAMBDA_SEM + LAMBDA_STR) - 1.0) < 1e-6, "λ1 + λ2 must equal 1"
