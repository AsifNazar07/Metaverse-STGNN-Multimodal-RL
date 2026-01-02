import os
import json
import numpy as np


class RLDataBuilder:

    def __init__(self, zones_config="../stgnn/k8s_autoscaler/zones_config.json"):

        with open(zones_config, "r") as f:
            self.zone_data = json.load(f)

        self.zones = list(self.zone_data.keys())
        self.num_zones = len(self.zones)

    # ----------------------------------------------------------------------
    # Normalize helper (for offline RL dataset)
    # ----------------------------------------------------------------------
    def normalize(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8)

    # ----------------------------------------------------------------------
    # Build single observation from stored zone data
    # ----------------------------------------------------------------------
    def build_obs(self, zone):
        z = self.zone_data[zone]

        # These are static placeholders unless updated elsewhere
        obs = [
            self.normalize(z["zone_features"][0], 0, 120),    # latency
            self.normalize(z["zone_features"][1], 0, 5),      # jitter
            self.normalize(z["zone_features"][2], 0, 0.2),    # packet loss
            self.normalize(z["active_users"], 0, 50),         # users
            self.normalize(np.mean(z["fused_embedding"]), 0, 1),  # embedding density
            0.5,  # placeholder CPU allocation
            0.5   # placeholder memory allocation
        ]
        return obs


    def generate_random_dataset(self, episodes=20000):

        dataset = {
            "states": [],
            "actions": [],
            "rewards": []
        }

        for _ in range(episodes):
            state = []
            actions = []
            rewards = []

            for zone in self.zones:
                state.extend(self.build_obs(zone))
                actions.append(np.random.uniform(-1, 1, size=3))  # random ΔCPU,ΔMem,ΔBW
                rewards.append(np.random.uniform(-5, 5))

            dataset["states"].append(state)
            dataset["actions"].append(actions)
            dataset["rewards"].append(sum(rewards))

        return dataset

    def save_dataset(self, dataset, path="offline_rl_dataset.npz"):
        np.savez_compressed(path, **dataset)
        print(f"[SAVED] Offline RL dataset → {path}")

    # ----------------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------------
    def load_dataset(self, path="offline_rl_dataset.npz"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No dataset found at {path}")

        data = np.load(path, allow_pickle=True)
        print(f"[LOADED] Offline RL dataset from {path}")
        return dict(data)
