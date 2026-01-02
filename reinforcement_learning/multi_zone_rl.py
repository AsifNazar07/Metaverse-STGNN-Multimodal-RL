import os
import numpy as np
import torch

from complex_simulation_env import MetaverseEnv
from models.ppo_agent import PPOAgent


# ------------------------------------------------------------------------------
# Setup Folders
# ------------------------------------------------------------------------------
os.makedirs("results/saved_models", exist_ok=True)
os.makedirs("results/training_logs", exist_ok=True)


# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------
STATE_DIM = 30 * 7   # 210
ACTION_DIM = 30 * 3  # 90

TOTAL_TRAINING_ITER = 2000
ROLLOUT_STEPS = 2048
SAVE_INTERVAL = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------
# Initialize Environment + Agent
# ------------------------------------------------------------------------------
env = MetaverseEnv()
agent = PPOAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    epochs=10,
    batch_size=64,
    device=DEVICE
)

print("\n[INFO] PPO Training Started")
print(f"[ENV]  Number of Zones: {env.num_zones}")
print(f"[RL]   Device: {DEVICE}")
print(f"[RL]   State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
print("------------------------------------------------------------------------------------")


# ------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------
reward_history = []

for iteration in range(1, TOTAL_TRAINING_ITER + 1):

    agent.train_step(env, rollout_steps=ROLLOUT_STEPS)

    # Quick evaluation roll
    state = env.reset()
    eval_reward = 0
    done = False
    steps = 0

    while not done and steps < 300:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            action, _ = agent.model.act(state_tensor)

        action_np = action.cpu().numpy().reshape(30, 3)
        next_state, reward, done, _ = env.step(action_np)

        eval_reward += reward
        steps += 1
        state = next_state

    reward_history.append(eval_reward)

    print(f"[ITER {iteration:04d}] Eval Reward: {eval_reward:.3f}")

    # Save model periodically
    if iteration % SAVE_INTERVAL == 0:
        save_path = f"results/saved_models/ppo_iter_{iteration}.pth"
        agent.save(save_path)
        print(f"[MODEL SAVED] → {save_path}")

        # Log reward history
        np.savetxt("results/training_logs/reward_history.txt", reward_history)


print("\n[TRAINING COMPLETED]")
print("Final model saved at results/saved_models/")
# Save final model
agent.save("results/saved_models/ppo_final.pth")