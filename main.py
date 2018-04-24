import gym


# could use the non-ram version, but this one is "tinier"
ROM = 'VideoPinball-ram-v0'

if __name__ == '__main__':
    env = gym.make(ROM)
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
