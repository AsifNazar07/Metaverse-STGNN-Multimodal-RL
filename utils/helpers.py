import json
import time
import os


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[SAVED] JSON → {path}")


class Timer:
    
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time


def ensure_dir(path):
    
    if not os.path.exists(path):
        os.makedirs(path)
