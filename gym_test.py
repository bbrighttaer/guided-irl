import gym

env = gym.make('CartPole-v1')
obs = env.reset()
print(f'observation.shape={env.observation_space.shape}, action.n={env.action_space.n}')
for i in range(100):
    env.render()
    action = env.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    if done:
        break
env.close()
