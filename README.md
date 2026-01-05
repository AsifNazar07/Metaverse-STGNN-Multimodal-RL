Metaverse-STGNN-Multimodal-RL

A Multimodal-Driven Spatio-Temporal GNN and Reinforcement Learning Framework for
Intent-Aware Resource Orchestration in Metaverse Environments

Overview

This repository provides the full experimental implementation of the framework proposed in our manuscript on multimodal, intent-aware resource orchestration for 6G Metaverse workloads.

The system integrates:

Multimodal intent extraction (vision, speech, motion)

Spatio-Temporal Graph Neural Networks (ST-GNN) for resource forecasting

Semantic–Structural Intent Verification (SSIV)

Intent-Based Networking (IBN) translation

Multi-Agent Reinforcement Learning (MARL) for closed-loop orchestration

Kubernetes-native enforcement with Prometheus/Grafana monitoring

The implementation is designed to demonstrate real-time, predictive, and intent-assured orchestration consistent with the methodology and evaluation reported in the paper.

Repository Structure
Metaverse-STGNN-Multimodal-RL/
│
├── README.md
├── requirements.txt
│
├── docs/
│   ├── architecture_overview.png
│   ├── stgnn_flow.png
│   ├── rl_system.png
│   ├── multimodal_pipeline.png
│   ├── k8s_prometheus_grafana.png
│   ├── screenshots/
│   └── reproducibility_guide.md
│
├── multimodal/
│   ├── image_captioning_BLIP_VOC2012/
│   ├── speech_recognition_Wav2Vec2_LibriSpeech/
│   └── motion_captioning_KAN_KITML/
│
├── stgnn/
│   ├── training/
│   ├── inference_api/
│   ├── k8s_autoscaler/
│   ├── visualization/
│   └── metrics_exporter.py
│
├── reinforcement_learning/
│   ├── RL_single_zone.py
│   ├── RL_multi_zone.py
│   └── rl_complex_environment.py
│
├── data/
└── utils/

Multimodal Encoders

Three pretrained modality-specific encoders are implemented, corresponding exactly to the datasets and formulations used in the paper.

Vision Encoder EBLIP(Iₜ)

Model: BLIP (Salesforce, image captioning base)

Dataset: PASCAL VOC 2012

Directory:
multimodal/image_captioning_BLIP_VOC2012/

Function: Converts visual scenes into semantic textual descriptions capturing user interaction context.

Audio Encoder EW2V2(Aₜ)

Model: Wav2Vec2 Base

Dataset: LibriSpeech 

Directory:
multimodal/speech_recognition_Wav2Vec2_LibriSpeech/

Function: Extracts robust speech representations for speech recognition and intent-related acoustic cues.

Motion Encoder EKAN(Mₜ)

Model: KAN-based motion-to-text encoder

Dataset: KIT Motion-Language (KIT-ML)

Directory:
multimodal/motion_captioning_KAN_KITML/

Function: Models nonlinear spatio-temporal motion and interaction semantics.

Multimodal Output

Each encoder produces a fixed-dimension embedding:

dᵥ: vision embedding dimension

dₐ: audio embedding dimension

dₘ: motion embedding dimension 

These embeddings are fused via late multimodal fusion and forwarded to the ST-GNN and SSIV modules.

Spatio-Temporal GNN (ST-GNN)

The ST-GNN predicts near-future resource demand for Metaverse orchestration zones:

CPU cores

Memory (MB)

Bandwidth (Mbps)

Latency (ms)

Key Components

training/stgnn_training_script.py offline training

inference_api/main.py — FastAPI inference server (/predict)

k8s_autoscaler/stgnn_k8s_controller.py — Kubernetes autoscaling logic

metrics_exporter.py — Prometheus metrics export

visualization/stgnn_live_visualization.py real-time simulation

ST-GNN Input
(zone_id, active_users, multimodal_intent_embedding)

Output
{CPU, Memory, Bandwidth, Latency} predictions

Kubernetes-Integrated Orchestration

A fully reproducible Kubernetes testbed is provided using Docker Desktop.

Features

ST-GNN FastAPI server for real-time inference

Python controller applying autoscaling via kubectl

Prometheus scraping prediction-driven metrics

Grafana dashboards for visualization

Kubernetes Manifests

stgnn-zones.yaml

prometheus-configmap.yaml

This establishes a closed-loop ML → prediction → orchestration → monitoring pipeline.

Reinforcement Learning Agents

The reinforcement_learning/ directory contains multiple RL configurations:

Implemented Agents

Single-Zone DQN (baseline)
RL_single_zone.py

Multi-Zone MARL (local + global rewards)
RL_multi_zone.py

Complex scenario with drift detection
rl_complex_environment.py

Optimized Objectives

Autoscaling decisions

Load balancing

Bandwidth allocation

Latency minimization

SLA adherence

RL agents consume ST-GNN predictions and verified intents to perform stable, closed-loop orchestration.

Live Simulation Environment

The live visualization renders:

Orchestration zones and nodes

Predicted resource trajectories

Growth/decay dynamics

Real-time API calls

Kubernetes updates 

This component is ideal for demonstrations and qualitative evaluation.

Installation & Execution
1. Create Environment
conda create -n metaverse python=3.10
conda activate metaverse

2. Install Dependencies
pip install -r requirements.txt

3. Start ST-GNN Inference API
uvicorn stgnn.inference_api.main:app --reload --host 0.0.0.0 --port 8000

4. Run Kubernetes Autoscaler
python stgnn/k8s_autoscaler/stgnn_k8s_controller.py

5. Launch Live Simulation
python stgnn/visualization/stgnn_live_visualization.py

Monitoring with Prometheus & Grafana
Prometheus
kubectl port-forward svc/prometheus-server 9090:80

Grafana
kubectl port-forward grafana-xxxx 3000:3000

Exported Metrics

stgnn_cpu_cores

stgnn_memory_mb

stgnn_bandwidth_mbps

stgnn_latency_ms

Reproducibility Notes

All datasets used are publicly available

Training and inference pipelines are separated

Experiments align with manuscript evaluation settings



