Metaverse-STGNN-Multimodal-RL

A Multimodal–Driven Spatio-Temporal GNN and Reinforcement Learning Framework for Metaverse Resource Orchestration

This repository contains the full experimental implementation from our research testbed integrating:

Multimodal encoders (Image → Text, Speech → Text, Motion → Text)

Spatio-Temporal Graph Neural Network (ST-GNN) for resource prediction

FastAPI online inference server

Kubernetes autoscaling pipeline (Docker Desktop K8s)

Prometheus + Grafana monitoring stack

Multi-Agent Reinforcement Learning (MARL) for closed-loop orchestration

Live ST-GNN-driven simulation environment

The goal of this system is to demonstrate a real-time, intent-aware, multimodal resource orchestration framework for metaverse environments, aligning with the methodology presented in the paper.

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
│   └── data/
│
├── reinforcement_learning/
└── utils/

1. Multimodal Encoders (EBLIP, EW2V2, EKAN)

This repository includes three pretrained encoders, each aligned with its corresponding dataset as described in the paper:

Image Encoder — EBLIP(It)

Model: BLIP (Salesforce, image captioning base)

Dataset: PASCAL VOC 2012

Directory: multimodal/image_captioning_BLIP_VOC2012/

Audio Encoder — EW2V2(At)

Model: Wav2Vec2 Base

Dataset: LibriSpeech train-clean-100

Directory: multimodal/speech_recognition_Wav2Vec2_LibriSpeech/

Motion Encoder — EKAN(Mt)

Model: KAN-based motion-to-text encoder

Dataset: KIT Motion-Language (KIT-ML)

Directory: multimodal/motion_captioning_KAN_KITML/

Each encoder produces a fixed-dimension embedding:

dv = image embedding dimension

da = audio embedding dimension

dm = motion embedding dimension (128-D as used in our experiments)

These embeddings feed the ST-GNN for resource forecasting.

2. Spatio-Temporal GNN (ST-GNN)

The ST-GNN predicts resource usage for metaverse zones:

CPU cores

Memory MB

Bandwidth Mbps

Latency ms

Components:

training/stgnn_training_script.py — Offline training

inference_api/main.py — FastAPI online server (/predict)

k8s_autoscaler/stgnn_k8s_controller.py — Periodic K8s autoscaling

metrics_exporter.py — Prometheus metrics producer

visualization/stgnn_live_visualization.py — Real-time evolving zone simulation

The ST-GNN receives:

(zone_name, active_users, multimodal embedding)


and predicts resource demands.

3. Kubernetes-Integrated Autoscaling

The repository includes a fully working K8s testbed built using Docker Desktop:

Features:

FastAPI ST-GNN server runs locally

Python controller triggers K8s autoscaling through kubectl set resources

Prometheus scrapes ST-GNN predictions as metrics

Grafana visualizes real-time performance

Kubernetes Files:

stgnn-zones.yaml

prometheus-configmap.yaml

This creates a reproducible orchestration loop connecting ML → prediction → autoscaling → monitoring.

4. Reinforcement Learning Agents

The reinforcement_learning/ directory contains:

Single-Zone DQN (baseline)

RL_single_zone.py

Multi-Zone MARL with global + local rewards

RL_multi_zone.py

Full complex scenario

rl_complex_environment.py

These agents learn:

Autoscaling

Load balancing

Bandwidth allocation

Latency minimization

RL integrates with the ST-GNN predictions to form a closed-loop controller.

5. Live Simulation Environment

stgnn_live_visualization.py renders:

Zones

Nodes

Live predicted resource values

Growth, decay, movement

Real-time API calls

Kubernetes updates (simulated)

This produces a continuous orchestration animation, ideal for demonstration or evaluation.

Installation
1. Create environment
conda create -n metaverse python=3.10
conda activate metaverse

2. Install dependencies
pip install -r requirements.txt

3. Start ST-GNN API
uvicorn stgnn.inference_api.main:app --reload --host 0.0.0.0 --port 8000

4. Run autoscaler
python stgnn/k8s_autoscaler/stgnn_k8s_controller.py

5. Start simulation
python stgnn/visualization/stgnn_live_visualization.py

6. Monitoring with Prometheus & Grafana

Port-forward Prometheus:

kubectl port-forward svc/prometheus-server 9090:80


Port-forward Grafana:

kubectl port-forward grafana-xxxx 3000:3000


Metric names available:

stgnn_cpu_cores

stgnn_memory_mb

stgnn_bandwidth_mbps

stgnn_latency_ms

7. Reproducibility for Reviewers

The docs/reproducibility_guide.md explains:

How to reproduce each multimodal encoder

How to run ST-GNN predictions

How to run Kubernetes autoscaling

How to run RL experiments

How to reproduce all figures from the paper

