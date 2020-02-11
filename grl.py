import pickle
import random
from collections import namedtuple

import gym
import numpy as np
import ptan
import torch
import torch.optim as optim
from ptan.agent import float32_preprocessor

from util import PGN, RewardNet

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4
DEMO_BATCH = 50
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action', 'reward', 'next_state'])
Trajectory = namedtuple('Trajectory', field_names=['prob', 'episode_steps'])


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


def process_demonstrations(demo_samples):
    traj_states, traj_actions, traj_qvals, traj_prob = [], [], [], []
    for traj in demo_samples:
        states, actions, rewards, qvals = [], [], [], []
        traj_prob.append(traj.prob)
        for step in traj.episode_steps:
            states.append(step.state)
            actions.append(step.action)
            rewards.append(step.reward)
        qvals.extend(calc_qvals(rewards))

        traj_states.append(states)
        traj_actions.append(actions)
        traj_qvals.append(qvals)
    traj_states = np.array(traj_states, dtype=np.object)
    traj_actions = np.array(traj_actions, dtype=np.object)
    traj_qvals = np.array(traj_qvals, dtype=np.object)
    traj_prob = np.array(traj_prob, dtype=np.float)
    return {'states': traj_states, 'actions': traj_actions, 'qvals': traj_qvals, 'traj_probs': traj_prob}


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent_net = PGN(env.observation_space.shape[0], env.action_space.n)
    reward_net = RewardNet(env.observation_space.shape[0] + 1)
    agent = ptan.agent.PolicyAgent(agent_net, preprocessor=float32_preprocessor, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    optimizer_agent = optim.Adam(agent_net.parameters(), lr=LEARNING_RATE)
    optimizer_reward = optim.Adam(reward_net.parameters(), lr=1e-2, weight_decay=1e-4)
    with open('demonstrations.list.pkl', 'rb') as f:
        demonstrations = pickle.load(f)
    assert (len(demonstrations) > DEMO_BATCH)
    print(f'Number of demonstrations: {len(demonstrations)}')
    demonstrations = process_demonstrations(demonstrations)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []
    loss_rwd = 0.

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        x = torch.cat([float32_preprocessor(exp.state), float32_preprocessor([int(exp.action)])]).view(1, -1)
        reward = reward_net(x)
        cur_rewards.append(reward.item())

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print(f'{step_idx}: reward: {reward:6.2f}, mean_100: {mean_rewards:6.2f}, '
                  f'episodes: {done_episodes}, reward function loss: {loss_rwd:6.4f}')
            if mean_rewards >= 500:
                print(f'Solved in {step_idx} steps and {done_episodes} episodes!')
                torch.save(agent_net.state_dict(), 'cartpole_learner.mod')
                torch.save(reward_net.state_dict(), 'cartpole-v1_reward_func.mod')
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        # reward function learning
        demo_states = demonstrations['states']
        demo_actions = demonstrations['actions']
        demo_probs = demonstrations['traj_probs']
        for rf_i in range(10):
            selected = np.random.choice(len(demonstrations), DEMO_BATCH)
            demo_states = demo_states[selected]
            demo_actions = demo_actions[selected]
            demo_probs = demo_probs[selected]
            demo_batch_states, demo_batch_actions = [], []
            for idx in range(len(demo_states)):
                demo_batch_states.extend(demo_states[idx])
                demo_batch_actions.extend(demo_actions[idx])
            demo_batch_states = torch.FloatTensor(demo_batch_states)
            demo_batch_actions = torch.FloatTensor(demo_batch_actions)
            D_demo = torch.cat([demo_batch_states, demo_batch_actions.view(-1, 1)], dim=-1)
            D_samp = torch.cat([states_v, batch_actions_t.float().view(-1, 1)], dim=-1)
            D_samp = torch.cat([D_demo, D_samp])
            # dummy importance weights - fix later
            z = torch.ones((D_samp.shape[0], 1))

            # objective
            D_demo_out = reward_net(D_demo)
            D_samp_out = reward_net(D_samp)
            D_samp_out = z * torch.exp(D_samp_out)
            loss_rwd = torch.mean(D_demo_out) - torch.log(torch.mean(D_samp_out))
            loss_rwd = -loss_rwd  # for maximization

            # update parameters
            optimizer_reward.zero_grad()
            loss_rwd.backward()
            optimizer_reward.step()

        # agent
        optimizer_agent.zero_grad()
        logits_v = agent_net(states_v)
        log_prob_v = torch.log_softmax(logits_v, dim=1)
        # REINFORCE
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer_agent.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()
    env.close()
