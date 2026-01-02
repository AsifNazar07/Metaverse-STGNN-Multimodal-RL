# k8s_autoscaler/metrics_exporter.py
from prometheus_client import Gauge, start_http_server

# ------------------------------------------------------------------------------
# Define Gauges
# ------------------------------------------------------------------------------
cpu_pred = Gauge("stgnn_cpu_prediction", "Predicted CPU usage %", ["zone"])
memory_pred = Gauge("stgnn_memory_prediction", "Predicted Memory usage %", ["zone"])
bandwidth_pred = Gauge("stgnn_bandwidth_prediction", "Predicted Bandwidth Mbps", ["zone"])
latency_pred = Gauge("stgnn_latency_prediction", "Predicted Latency ms", ["zone"])
allocated_cpu = Gauge("stgnn_allocated_cpu_millicores",
                      "Allocated CPU in milli-cores",
                      ["zone"])

# ------------------------------------------------------------------------------
# Update functions (CALLED FROM AUTOSCALER)
# ------------------------------------------------------------------------------
def update_zone_predictions(zone, cpu, memory, bandwidth, latency):
    cpu_pred.labels(zone).set(cpu)
    memory_pred.labels(zone).set(memory)
    bandwidth_pred.labels(zone).set(bandwidth)
    latency_pred.labels(zone).set(latency)

def update_zone_allocations(zone, cpu_alloc_m):
    allocated_cpu.labels(zone).set(cpu_alloc_m)


# ------------------------------------------------------------------------------
# Exporter entry point
# ------------------------------------------------------------------------------
def start_exporter(port=9105):
    print(f"[METRICS] Prometheus exporter running on port {port} ...")
    start_http_server(port)


if __name__ == "__main__":
    start_exporter()
    print("[INFO] Metrics exporter running forever...")
    import time
    while True:
        time.sleep(1)
