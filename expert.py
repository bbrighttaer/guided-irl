from collections import namedtuple
import pickle
import gym
import ptan
from ptan.agent import float32_preprocessor
import torch
import numpy as np

from util import PGN

GAMMA = 0.99
NUM_TRAJS = 100

EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action', 'reward', 'next_state'])
Trajectory = namedtuple('Trajectory', field_names=['prob', 'episode_steps'])

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load('cartpole_expert.mod'))
    net.eval()
    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, preprocessor=float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    trajectories = []
    for ep in range(NUM_TRAJS):
        episode = []
        qt = 1.0
        for step_idx, exp in enumerate(exp_source):
            probs = torch.softmax(net(float32_preprocessor(exp.state).view(1, -1)), dim=1)
            probs = probs.squeeze()[int(exp.action)].item()
            qt *= probs
            episode.append(EpisodeStep(state=exp.state, action=int(exp.action), reward=exp.reward,
                                       next_state=exp.last_state))
            if exp.last_state is None:
                break
        print(np.prod())
        trajectories.append(Trajectory(prob=qt, episode_steps=episode))
        print(f'Number of trajectories: {len(trajectories)}')
    with open('demonstrations.list.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    env.close()
