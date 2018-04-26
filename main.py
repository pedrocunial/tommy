import gym


# could use the non-ram version, but this one is "tinier"
PINBALL = 'VideoPinball-ram-v0'
CART_POLE = 'CartPole-v0'
ROM = PINBALL

if __name__ == '__main__':
    env = gym.make(ROM)
    highscore = 0
    for i_episode in range(20):
        obs = env.reset()
        points = 0
        for t in range(1000):
            env.render()
            action = env.action_space.sample()  # randomic action
            obs, rwd, done, info = env.step(action)
            points += rwd
            if done:
                if points > highscore:
                    highscore = points
                    print('A new highscore! {}'.format(highscore))
                print('Done after {} trials'.format(t + 1))
                break
