import gym
import numpy as np
import ptan
import torch
import torch.optim as optim
import random

from util import PGN

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

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
            print(f'{step_idx}: reward: {reward:6.2f}, mean_100: {mean_rewards:6.2f}, episodes: {done_episodes}')
            if mean_rewards >= 500:
                print(f'Solved in {step_idx} steps and {done_episodes} episodes!')
                torch.save(net.state_dict(), 'cartpole_expert.mod')
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = net(states_v)
        log_prob_v = torch.log_softmax(logits_v, dim=1)
        # REINFORCE
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()
    env.close()
