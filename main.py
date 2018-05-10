import gym

from numpy import cumsum
from numpy.random import choice
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import Adam


PINBALL = 'VideoPinball-v0'
CART_POLE = 'CartPole-v0'
ROM = PINBALL

# NOTE: actions represent the 9 possible positions for the joystick
#       in Géron p470
LAUNCH_BALL = 1
BUMPER_BOTH = 2
BUMPER_RIGHT = 3
BUMPER_LEFT = 4
PULL_BALL = 5
BUMPER_BOTH2 = 6
BUMPER_RIGHT2 = 7
BUMPER_LEFT2 = 8

LAUNCH_STEPS = 20  # frames required to launch ball


def preprocess_observation(obs):
    # from Géron p470
    # my image is (250, 160, 3) -- for the raw version of VideoPinball
    img = obs[20:200:2, ::2]
    img = img.mean(axis=2)
    img = (img - 128) / 128 - 1  # normalize
    return img.reshape(88, 80, 1)


def max_index(options):
    # print(options)
    # cs = cumsum(options)
    # res = [i / cs for i in options]
    # print(len(res), res)
    options = options[0]
    cs = cumsum(options)
    probas = [x / cs for x in options]
    print(probas, options, cs)
    return choice(9, p=probas)


def start_game(env):
    print('Start game! Pulling Ball!')
    for i in range(LAUNCH_STEPS):
        env.render()
        env.step(PULL_BALL)
    print('Ball pulled! Launching!')
    for i in range(LAUNCH_STEPS):
        env.render()
        env.step(LAUNCH_BALL)


if __name__ == '__main__':
    env = gym.make(ROM)
    epochs = 1000
    highscore = 0

    # using model from
    # https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    model = Sequential()
    # model.add(Permute((2, 3, 1), input_shape=env.observation_space.shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4),
                     input_shape=env.observation_space.shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))  # 9 possible actions
    model.add(Activation('sigmoid'))
    print(model.summary())

    # lr value from
    # https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    model.compile(loss='mse', optimizer=Adam(lr=.00025))

    for i_episode in range(20):
        obs = env.reset()
        points = 0
        action = PULL_BALL
        for t in range(epochs):
            done = False
            iters = LAUNCH_STEPS
            start_game(env)
            while not done:
                env.render()
                obs, rwd, done, info = env.step(action)
                obs = obs.reshape(-1, obs.shape[0], obs.shape[1], obs.shape[2])
                # predict action from observed
                action = max_index(model.predict(obs))
                print(action)
                points += rwd
                iters += 1
                if done:
                    if points > highscore:
                        highscore = points
                        print('A new highscore! {}'.format(highscore))
                    print('Done after {} trials'.format(iters))
