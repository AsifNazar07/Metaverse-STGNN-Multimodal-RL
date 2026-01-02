"""
====================================================================================
FastAPI Inference Server for ST-GNN
------------------------------------------------------------------------------------
This server provides a production-ready API endpoint for online inference:

POST /predict
{
    "fused_embedding": [...],
    "active_users": 32,
    "zone_features": [latency, jitter, packet_loss, movement_rate]
}

Response:
{
    "cpu": float,
    "memory": float,
    "bandwidth": float,
    "latency": float
}

This service is used by:
    • Kubernetes Autoscaler
    • Visualization Engine
    • Experiments & evaluation scripts
====================================================================================
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from stgnn_inference import stgnn_predict

app = FastAPI(
    title="ST-GNN Inference API",
    description="Online prediction service for metaverse resource forecasting.",
    version="1.0"
)

# ==============================================================================
# 1. Request Schema
# ==============================================================================
class PredictionRequest(BaseModel):
    fused_embedding: list = Field(..., description="1024-D fused embedding")
    active_users: int = Field(..., description="Number of active users in zone")
    zone_features: list = Field(..., description="[latency, jitter, packet_loss, movement]")


# ==============================================================================
# 2. Prediction Endpoint
# ==============================================================================
@app.post("/predict")
def predict(req: PredictionRequest):
    result = stgnn_predict(
        fused_embedding=req.fused_embedding,
        active_users=req.active_users,
        zone_features=req.zone_features
    )
    return result


# ==============================================================================
# 3. Run Command (for docs)
# ==============================================================================
"""
To run this server:

uvicorn main:app --reload --host 0.0.0.0 --port 8000

Test using:

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"fused_embedding":[0.1,0.3,...], "active_users":10,
          "zone_features":[10,0.3,0.01,5]}'
"""
