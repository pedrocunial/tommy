import gym


# could use the non-ram version, but this one is "tinier"
PINBALL = 'VirtualPinball-ram-v0'
ROM = 'CartPole-v0'

if __name__ == '__main__':
    env = gym.make(ROM)
    highscore = 0
    for i_episode in range(20):
        obs = env.reset()
        points = 0
        for t in range(1000):
            env.render()
            # if angle is pos, move right else left
            action = 1 if obs[2] > 0 else 0
            obs, rwd, done, info = env.step(action)
            points += rwd
            if done:
                if points > highscore:
                    highscore = points
                break
