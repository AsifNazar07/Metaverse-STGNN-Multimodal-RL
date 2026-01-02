import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        """
        state_dim   = 210  (30 zones × 7 features)
        action_dim  = 90   (3 actions × 30 zones)
        """
        super(ActorCritic, self).__init__()

        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor head → outputs continuous action logits
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """
        Returns:
            action_mean : μ(s)
            action_std  : σ (learnable)
            value       : V(s)
        """
        x = self.base(state)

        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd)

        value = self.critic(x)

        return action_mean, action_std, value

    # Used during PPO rollout
    def act(self, state):
        """
        Samples action from actor distribution.
        """
        mean, std, _ = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        return action, logprob

    # Used during PPO updates
    def evaluate(self, state, action):
        """
        Computes logprob, entropy, and value for PPO update step.
        """
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return logprob, entropy, value
