import time
import json
import random
import requests
import subprocess

# Import metrics update functions
from metrics_exporter import (
    update_zone_predictions,
    update_zone_allocations
)

FASTAPI_URL = "http://localhost:8000/predict"
INTERVAL = 10

CPU_UPPER = 75
CPU_LOWER = 25
MIN_CPU = 100
MAX_CPU = 1200


# ------------------------------------------------------------------------------
# Load 30-zone configuration
# ------------------------------------------------------------------------------
def load_zone_config():
    with open("zones_config.json", "r") as f:
        return json.load(f)

zones = load_zone_config()
ZONES = list(zones.keys())


# ------------------------------------------------------------------------------
# K8s update
# ------------------------------------------------------------------------------
def kubectl_set_resources(zone, cpu_millicores):
    cpu_str = f"{cpu_millicores}m"
    cmd = [
        "kubectl", "set", "resources", "deployment", zone,
        f"--requests=cpu={cpu_str}", f"--limits=cpu={cpu_str}"
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[K8S] Zone {zone}: CPU updated to {cpu_str}")
    except Exception as e:
        print(f"[K8S ERROR] {zone}: {e}")


# ------------------------------------------------------------------------------
# Dynamic feature update
# ------------------------------------------------------------------------------
def update_zone_features(zone):
    zone["zone_features"][0] += random.uniform(-3, 3)
    zone["zone_features"][1] += random.uniform(-0.1, 0.1)
    zone["zone_features"][2] += random.uniform(-0.01, 0.01)
    zone["zone_features"][3] += random.uniform(-2, 2)

    zone["zone_features"][0] = max(5, min(120, zone["zone_features"][0]))
    zone["zone_features"][1] = max(0, min(5, zone["zone_features"][1]))
    zone["zone_features"][2] = max(0, min(0.2, zone["zone_features"][2]))
    zone["zone_features"][3] = max(0, min(50, zone["zone_features"][3]))

    return zone


# ------------------------------------------------------------------------------
# Scaling rule
# ------------------------------------------------------------------------------
def compute_new_cpu(pred_cpu, current_cpu):
    if pred_cpu > CPU_UPPER:
        current_cpu += 150
    elif pred_cpu < CPU_LOWER:
        current_cpu -= 100
    return max(MIN_CPU, min(MAX_CPU, current_cpu))


# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
def autoscale_loop():
    print("[INFO] Autoscaler running with Prometheus metrics integration.\n")

    cpu_alloc = {zone: 300 for zone in ZONES}

    while True:
        for zone in ZONES:

            zone_data = update_zone_features(zones[zone])

            payload = {
                "fused_embedding": zone_data["fused_embedding"],
                "active_users": zone_data["active_users"],
                "zone_features": zone_data["zone_features"]
            }

            # ST-GNN Prediction
            response = requests.post(FASTAPI_URL, json=payload)
            pred = response.json()

            pred_cpu = pred["cpu"]
            pred_memory = pred["memory"]
            pred_bandwidth = pred["bandwidth"]
            pred_latency = pred["latency"]

            old_cpu = cpu_alloc[zone]
            new_cpu = compute_new_cpu(pred_cpu, old_cpu)

            # ------------------------
            # Update Prometheus metrics
            # ------------------------
            update_zone_predictions(
                zone,
                pred_cpu,
                pred_memory,
                pred_bandwidth,
                pred_latency
            )

            update_zone_allocations(zone, old_cpu)

            # ------------------------
            # Kubernetes scaling
            # ------------------------
            if new_cpu != old_cpu:
                kubectl_set_resources(zone, new_cpu)
                cpu_alloc[zone] = new_cpu

            print(f"[ZONE {zone}] CPU Pred={pred_cpu:.2f} | Current={old_cpu}m | New={new_cpu}m")

        time.sleep(INTERVAL)


# ------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    autoscale_loop()
