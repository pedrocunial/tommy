import gym
import numpy as np

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, LSTM, Permute
from keras.optimizers import Adam
import keras.backend as K
# from collections import deque

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


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
TIME_DEPTH = 5


# def preprocess_observation(obs):
#     # from Géron p470
#     # my image is (250, 160, 3) -- for the raw version of VideoPinball
#     img = obs[20:200:2, ::2]
#     img = img.mean(axis=2)
#     img = (img - 128) / 128 - 1  # normalize
#     return img.reshape(88, 80, 1)


# def max_index(options):
#     # print(options)
#     # cs = cumsum(options)
#     # res = [i / cs for i in options]
#     # print(len(res), res)
#     options = options[0]
#     cs = cumsum(options)
#     probas = [x / cs for x in options]
#     print(probas, options, cs)
#     return choice(9, p=probas)


# def start_game(env, frames):
#     print('Start game! Pulling Ball!')
#     for i in range(LAUNCH_STEPS):
#         env.render()
#         obs, _, _, _ = env.step(PULL_BALL)
#         frames.append(obs)
#     print('Ball pulled! Launching!')
#     for i in range(LAUNCH_STEPS):
#         env.render()
#         obs, _, _, _ = env.step(LAUNCH_BALL)
#         frames.append(obs)

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    '''
    from keras-kl example
    https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    '''
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        '''
        We could perform this processing step in `process_observation`.
        In this case, however, we would need to store a `float32` array
        instead, which is 4x more memory intensive than an `uint8` array.
        This matters if we store 1M observations.
        '''
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


if __name__ == '__main__':
    env = gym.make(ROM)
    epochs = 1000
    highscore = 0
    shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    print(shape)

    # using model from
    # https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    # but added a rnn for time
    model = Sequential()
    # model.add(Permute((2, 3, 1), input_shape=env.observation_space.shape))
    # observation_space.shape + (5,) to add 5 time frames (last 5 frames)
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')

    model.add(Conv2D(32, (8, 8), strides=(4, 4), name='conv0_open_layer'))
    model.add(Activation('relu', name='relu0'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), name='conv1_4x4_stride_2x2'))
    model.add(Activation('relu', name='relu1'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), name='conv2_3x3_nostride'))
    model.add(Activation('relu', name='relu2'))
    # default activation is tanh
    # model.add(LSTM(64, activation='tanh', name='lstm'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, name='dense0'))
    model.add(Activation('relu', name='relu3'))
    # 9 possible actions
    model.add(Dense(env.action_space.n, name='dense1_final_dense'))
    # using sigmoid as sugested by the professor
    model.add(Activation('sigmoid', name='output_sigmoid'))
    print(model.summary())

    # lr value from
    # https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    model.compile(loss='mse', optimizer=Adam(lr=.00025))

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy,
                   memory=memory, processor=processor, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000, train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    # Okay, now it's time to learn something! We capture the interrupt
    # exception so that training can be prematurely aborted. Notice that
    # you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(ROM)
    checkpoint_weights_filename = 'dqn_' + ROM + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ROM)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                         interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
    print('done')

    # for i_episode in range(20):
    #     print('started episode #{}'.format(i_episode))
    #     obs = env.reset()
    #     points = 0
    #     action = PULL_BALL
    #     for t in range(epochs):
    #         done = False
    #         iters = LAUNCH_STEPS
    #         frames = deque([0] * TIME_DEPTH, TIME_DEPTH)  # keeps size
    #         while not done:
    #             env.render()
    #             obs, rwd, done, info = env.step(action)
    #             print(obs)
    #             frames.append(obs)  # deque keeps its size
    #             # obs = obs.reshape(-1, obs.shape[0], obs.shape[1], obs.shape[2])
    #             # predict action from observed
    #             # action = ()
    #             print(action)
    #             points += rwd
    #             iters += 1
    #             if done:
    #                 if points > highscore:
    #                     highscore = points
    #                     print('A new highscore! {}'.format(highscore))
    #                 print('Done after {} trials'.format(iters))
