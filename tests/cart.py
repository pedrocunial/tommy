import gym
import numpy as np

CART_ROM = 'CartPole-v0'
N_EPISODES = 15
LEN_EPISODE = 1000

def basic_policy(obs):
    # obs[2] = angle left
    return 0 if obs[2] < 0 else 1

if __name__ == '__main__':
    env = gym.make(CART_ROM)
    totals = []
    for episode in range(N_EPISODES):
        episode_rewards = 0
        obs = env.reset()
        for step in range(LEN_EPISODE):  # better than inf loop
            action = basic_policy(obs)
            obs, reward, done, info = env.step(action)
            img = env.render(mode='rgb_array')
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)
    env.close()
    print('mean: {} | stddev: {} | min: {} | max: {}'.format(
        np.mean(totals), np.std(totals), np.min(totals), np.max(totals)
    ))
