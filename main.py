import gym


PINBALL = 'VideoPinball-v0'
CART_POLE = 'CartPole-v0'
ROM = PINBALL

# NOTE: actions represent the 9 possible positions for the joystick
#       in Géron p470
BUMPER_BOTH = 2
BUMPER_RIGHT = 3
BUMPER_LEFT = 4
PULL_BALL = 5
BUMPER_BOTH2 = 6
BUMPER_RIGHT2 = 7
BUMPER_LEFT2 = 8

def preprocess_observation(obs):
    # from Géron p470
    # my image is (250, 160, 3) -- for the raw version of VideoPinball
    img = obs[20:200:2, ::2]
    img = img.mean(axis=2)
    img = (img - 128) / 128 - 1  # normalize
    return img.reshape(88, 80, 1)

if __name__ == '__main__':
    env = gym.make(ROM)
    highscore = 0
    for i_episode in range(20):
        obs = env.reset()
        points = 0
        action = 8
        for t in range(1000):
            env.render()
            action = 4
            print(action)
            obs, rwd, done, info = env.step(action)
            points += rwd
            if done:
                if points > highscore:
                    highscore = points
                    print('A new highscore! {}'.format(highscore))
                print('Done after {} trials'.format(t + 1))
                break
