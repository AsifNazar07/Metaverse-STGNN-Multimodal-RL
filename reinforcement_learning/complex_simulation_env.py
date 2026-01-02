import numpy as np
import json
import requests
import random


FASTAPI_URL = "http://localhost:8000/predict"


class MetaverseEnv:

    def __init__(self, zones_config_path="../stgnn/k8s_autoscaler/zones_config.json"):

        with open(zones_config_path, "r") as f:
            self.zone_data = json.load(f)

        self.zones = list(self.zone_data.keys())
        self.num_zones = len(self.zones)

        # RL Action space per zone: ΔCPU, ΔMEM, ΔBW
        #  -1 = decrease
        #   0 = maintain
        #  +1 = increase
        self.action_space = 3

        # Normalize allocations (initial)
        self.cpu_alloc = {z: 0.3 for z in self.zones}   # in cores (0.1 to 2.0)
        self.mem_alloc = {z: 512 for z in self.zones}   # in MB (128 to 2048)
        self.bw_alloc = {z: 20 for z in self.zones}     # in Mbps (5 to 200)

        # RL Observation vector per zone
        # [cpu_pred, mem_pred, bw_pred, latency, cpu_alloc, mem_alloc, bw_alloc]
        self.obs_dim = 7
        self.state_dim = self.num_zones * self.obs_dim

        self.max_steps = 20000
        self.current_step = 0

    # --------------------------------------------------------------------------
    # Query ST-GNN for predictions
    # --------------------------------------------------------------------------
    def get_prediction(self, zone):
        z = self.zone_data[zone]

        payload = {
            "fused_embedding": z["fused_embedding"],
            "active_users": z["active_users"],
            "zone_features": z["zone_features"]
        }

        try:
            res = requests.post(FASTAPI_URL, json=payload).json()
            return res  # cpu, memory, bandwidth, latency
        except:
            return {"cpu": 0, "memory": 0, "bandwidth": 0, "latency": 0}

    # --------------------------------------------------------------------------
    # Apply RL actions (ΔCPU, ΔMEM, ΔBW)
    # --------------------------------------------------------------------------
    def apply_action(self, zone, action):
        cpu_change = action[0]     # -1, 0, +1
        mem_change = action[1]
        bw_change = action[2]

        # CPU: 0.1 → 2.0 cores
        self.cpu_alloc[zone] = np.clip(self.cpu_alloc[zone] + cpu_change * 0.1, 0.1, 2.0)

        # MEMORY: 128MB → 2048MB
        self.mem_alloc[zone] = np.clip(self.mem_alloc[zone] + mem_change * 128,
                                       128, 2048)

        # BANDWIDTH: 5Mbps → 200Mbps
        self.bw_alloc[zone] = np.clip(self.bw_alloc[zone] + bw_change * 5,
                                      5, 200)

    # --------------------------------------------------------------------------
    # Reward Function (multi-objective)
    # --------------------------------------------------------------------------
    def compute_reward(self, pred, zone):
        cpu_pred = pred["cpu"]
        mem_pred = pred["memory"]
        bw_pred = pred["bandwidth"]
        latency = pred["latency"]

        cpu_a = self.cpu_alloc[zone]
        mem_a = self.mem_alloc[zone]
        bw_a = self.bw_alloc[zone]

        # Penalize overload
        cpu_penalty = max(0, cpu_pred - cpu_a * 100)
        mem_penalty = max(0, mem_pred - (mem_a / 2048) * 100)
        bw_penalty = max(0, bw_pred - bw_a)

        # Latency cost (QoS)
        latency_penalty = latency / 10

        # Provisioning cost (normalized)
        cpu_cost = cpu_a * 0.1
        mem_cost = mem_a / 4096
        bw_cost = bw_a / 200

        reward = (
            -cpu_penalty * 2
            -mem_penalty
            -bw_penalty
            -latency_penalty * 1.5
            -(cpu_cost + mem_cost + bw_cost)
        )

        return reward

    # --------------------------------------------------------------------------
    # Build observation vector
    # --------------------------------------------------------------------------
    def get_obs(self):
        obs = []
        for z in self.zones:
            pred = self.get_prediction(z)

            obs_zone = [
                pred["cpu"] / 100,
                pred["memory"] / 100,
                pred["bandwidth"] / 1000,
                pred["latency"] / 200,
                self.cpu_alloc[z] / 2.0,
                self.mem_alloc[z] / 2048,
                self.bw_alloc[z] / 200
            ]
            obs.extend(obs_zone)
        return np.array(obs, dtype=np.float32)

    # --------------------------------------------------------------------------
    # Step Function
    # --------------------------------------------------------------------------
    def step(self, actions):
        total_reward = 0

        for idx, zone in enumerate(self.zones):
            action = actions[idx]
            self.apply_action(zone, action)

            pred = self.get_prediction(zone)
            reward = self.compute_reward(pred, zone)

            total_reward += reward

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self.get_obs()

        return next_state, total_reward, done, {}

    # --------------------------------------------------------------------------
    # Reset Environment
    # --------------------------------------------------------------------------
    def reset(self):
        self.current_step = 0
        return self.get_obs()
