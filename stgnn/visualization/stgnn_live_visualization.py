import time
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FASTAPI_URL = "http://localhost:8000/predict"
PROMETHEUS_URL = "http://localhost:9105/metrics"

# Load zone list
with open("../k8s_autoscaler/zones_config.json", "r") as f:
    ZONES = list(json.load(f).keys())

NUM_ZONES = len(ZONES)


# ------------------------------------------------------------------------------
# Parse Prometheus Metrics
# ------------------------------------------------------------------------------
def get_prometheus_allocations():
    try:
        res = requests.get(PROMETHEUS_URL).text
        cpu_alloc = {}

        for line in res.split("\n"):
            if line.startswith("stgnn_allocated_cpu_millicores"):
                # Example:
                # stgnn_allocated_cpu_millicores{zone="zone-1"} 300
                try:
                    zone = line.split('zone="')[1].split('"')[0]
                    value = float(line.split("} ")[1])
                    cpu_alloc[zone] = value
                except:
                    pass

        return cpu_alloc

    except Exception as e:
        print(f"[ERROR] Prometheus fetch failed: {e}")
        return {zone: 0 for zone in ZONES}


# ------------------------------------------------------------------------------
# Query ST-GNN Predictor
# ------------------------------------------------------------------------------
def get_zone_prediction(embedding, users, features):
    payload = {
        "fused_embedding": embedding,
        "active_users": users,
        "zone_features": features
    }
    try:
        res = requests.post(FASTAPI_URL, json=payload).json()
        return res  # {cpu, memory, bandwidth, latency}
    except:
        return {"cpu": 0, "memory": 0, "bandwidth": 0, "latency": 0}


# ------------------------------------------------------------------------------
# Load initial zone dataset
# ------------------------------------------------------------------------------
with open("../k8s_autoscaler/zones_config.json", "r") as f:
    zone_data = json.load(f)


# ------------------------------------------------------------------------------
# Visualization Setup
# ------------------------------------------------------------------------------
plt.style.use("dark_background")

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

cpu_ax = axs[0, 0]
mem_ax = axs[0, 1]
bw_ax = axs[1, 0]
heat_ax = axs[1, 1]

cpu_ax.set_title("CPU Predictions (All Zones)")
mem_ax.set_title("Memory Predictions (All Zones)")
bw_ax.set_title("Bandwidth Predictions (All Zones)")
heat_ax.set_title("CPU Heatmap (30 Zones)")


cpu_lines = {}
mem_lines = {}
bw_lines = {}

x_vals = list(range(100))
history = {
    "cpu": {zone: [0]*100 for zone in ZONES},
    "memory": {zone: [0]*100 for zone in ZONES},
    "bandwidth": {zone: [0]*100 for zone in ZONES},
}


# Create line objects
for zone in ZONES:
    (cpu_line,) = cpu_ax.plot(x_vals, history["cpu"][zone], label=zone)
    (mem_line,) = mem_ax.plot(x_vals, history["memory"][zone], label=zone)
    (bw_line,) = bw_ax.plot(x_vals, history["bandwidth"][zone], label=zone)
    cpu_lines[zone] = cpu_line
    mem_lines[zone] = mem_line
    bw_lines[zone] = bw_line

cpu_ax.legend(fontsize=6, ncol=5)
mem_ax.legend(fontsize=6, ncol=5)
bw_ax.legend(fontsize=6, ncol=5)


# ------------------------------------------------------------------------------
# Update Loop
# ------------------------------------------------------------------------------
def update(frame):
    global zone_data

    cpu_matrix = []

    for zone in ZONES:
        z = zone_data[zone]

        pred = get_zone_prediction(
            z["fused_embedding"],
            z["active_users"],
            z["zone_features"]
        )

        history["cpu"][zone].append(pred["cpu"])
        history["cpu"][zone].pop(0)

        history["memory"][zone].append(pred["memory"])
        history["memory"][zone].pop(0)

        history["bandwidth"][zone].append(pred["bandwidth"])
        history["bandwidth"][zone].pop(0)

        cpu_lines[zone].set_ydata(history["cpu"][zone])
        mem_lines[zone].set_ydata(history["memory"][zone])
        bw_lines[zone].set_ydata(history["bandwidth"][zone])

        cpu_matrix.append(pred["cpu"])

    # Heatmap
    heat_ax.clear()
    heat_ax.set_title("CPU Heatmap (30 Zones)")
    heatmap_data = np.array(cpu_matrix).reshape(5, 6)  # 30 zones → grid 5x6
    heat_ax.imshow(heatmap_data, cmap="inferno")
    for i in range(5):
        for j in range(6):
            zone_name = ZONES[i*6+j]
            heat_ax.text(j, i, zone_name, ha="center", va="center", fontsize=7)

    return []


ani = FuncAnimation(fig, update, interval=3000)

plt.tight_layout()
plt.show()
