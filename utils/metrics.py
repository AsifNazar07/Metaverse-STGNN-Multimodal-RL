
import numpy as np


def mse(pred, true):
    return np.mean((np.array(pred) - np.array(true)) ** 2)


def rmse(pred, true):
    return np.sqrt(mse(pred, true))


def mae(pred, true):
    return np.mean(np.abs(np.array(pred) - np.array(true)))


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def qos_penalty(cpu_pred, cpu_alloc, mem_pred, mem_alloc, bw_pred, bw_alloc):
    """
    Multi-objective QoS penalty used optionally for evaluation:
        - penalizes CPU, memory, and bandwidth insufficiency
    """

    cpu_p = max(0, cpu_pred - cpu_alloc * 100)
    mem_p = max(0, mem_pred - (mem_alloc / 2048) * 100)
    bw_p = max(0, bw_pred - bw_alloc)

    return cpu_p + mem_p + bw_p
