import numpy as np
import ptan
import torch
from torch import nn as nn


class Agent(ptan.agent.PolicyAgent):

    @torch.no_grad()
    def __call__(self, states):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
                if states.ndim == 1:
                    states = states.view(1, -1)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = torch.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions)


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class RewardNet(nn.Module):
    def __init__(self, input_size):
        super(RewardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
