import matplotlib.pyplot as plt
import numpy as np


def plot_reward_curve(reward_history, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(reward_history, color="cyan")
    plt.title("RL Reward Curve (PPO Training)")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_heatmap(values, title="Heatmap", size=(5, 6)):
  
    matrix = np.array(values).reshape(size)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="inferno")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses, label="Loss", save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, label=label)
    plt.title(f"{label} Curve")
    plt.xlabel("Step")
    plt.ylabel(label)
    plt.grid()

    if save_path:
        plt.savefig(save_path)
    plt.legend()
    plt.show()
def plot_multiple_curves(curves_dict, title="Training Metrics", save_path=None):
    plt.figure(figsize=(10, 6))
    for label, values in curves_dict.items():
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()  