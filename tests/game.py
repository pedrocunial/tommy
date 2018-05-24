import gym


if __name__ == '__main__':
    env = gym.make('AirRaid-v0')

    env.seed(0)
    env.reset()

    for i in range(10000):
        env.render()
        obs, rew, done, _ = env.step(env.action_space.sample())
        if done:
            break

    env.close()
