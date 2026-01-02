import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .actor_critic import ActorCritic


class PPOAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=64,
        device="cpu"
    ):
        self.device = device

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

    # ----------------------------------------------------------------------
    # Rollout: collect transitions from environment
    # ----------------------------------------------------------------------
    def rollout(self, env, steps):
        states = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []

        state = env.reset()

        for _ in range(steps):
            state_t = torch.tensor(state, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                action, logprob = self.model.act(state_t)
                _, _, value = self.model.forward(state_t)

            action_np = action.cpu().numpy()
            logprob_np = logprob.cpu().numpy()
            value_np = value.cpu().numpy()

            next_state, reward, done, _ = env.step(self.reshape_actions(action_np))

            states.append(state)
            actions.append(action_np)
            logprobs.append(logprob_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value_np)

            state = next_state

            if done:
                state = env.reset()

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.float32).to(self.device),
            torch.tensor(logprobs, dtype=torch.float32).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
            torch.tensor(values, dtype=torch.float32).to(self.device)
        )

    # Convert 90 continuous actions → 30 (ΔCPU, ΔMEM, ΔBW)
    def reshape_actions(self, action_vec):
        return action_vec.reshape(30, 3)

    # ----------------------------------------------------------------------
    # Compute advantages using GAE
    # ----------------------------------------------------------------------
    def compute_advantages(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ----------------------------------------------------------------------
    # PPO Update Step
    # ----------------------------------------------------------------------
    def ppo_update(self, states, actions, old_logprobs, returns, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        for _ in range(self.epochs):
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size

                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_old_logprobs = old_logprobs[start:end]
                batch_returns = returns[start:end]
                batch_advantages = advantages[start:end]

                # Evaluate actions under current policy
                logprobs, entropy, values = self.model.evaluate(batch_states, batch_actions)

                ratio = torch.exp(logprobs - batch_old_logprobs)

                # PPO clipped objective
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                entropy_bonus = entropy.mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # ----------------------------------------------------------------------
    # Full training step for PPO
    # ----------------------------------------------------------------------
    def train_step(self, env, rollout_steps=2048):
        states, actions, logprobs, rewards, dones, values = self.rollout(env, rollout_steps)
        advantages, returns = self.compute_advantages(rewards, values, dones)
        self.ppo_update(states, actions, logprobs, returns, advantages)

    # ----------------------------------------------------------------------
    # Save model
    # ----------------------------------------------------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # ----------------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------------
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
