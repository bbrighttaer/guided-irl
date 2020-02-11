import gym
import torch
import ptan

from util import Agent, PGN

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load('cartpole_expert.mod'))
    net.eval()
    agent = Agent(net, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    state = env.reset()
    for _ in range(1000):
        env.render()
        action = agent(state)
        new_state, reward, done, _ = env.step(int(action))
        print(f'reward {reward}')
        if done:
            print('Resetting environment...')
            state = env.reset()
    env.close()
