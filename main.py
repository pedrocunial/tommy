import gym
import numpy as np
import argparse

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation
from keras.layers import Conv2D, Flatten, Permute
from keras.optimizers import Adam

# keras-extra lib from
# https://github.com/anayebi/keras-extra/blob/master/extra.py
# from lib.extra import TimeDistributedConvolution2D, TimeDistributedFlatten
# from collections import deque

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


PINBALL = 'VideoPinball-v0'
CART_POLE = 'CartPole-v0'
BREAKOUT = 'BreakoutDeterministic-v4'
AIRRAID = 'AirRaid-v0'   # space-invaders like game
ROM = BREAKOUT

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


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 3  # time spam
# TIME_SPAM = 5


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('--steps', type=int, default=10000)

    args = parser.parse_args()

    env = gym.make(ROM)
    epochs = 1000
    highscore = 0
    shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    print(shape)

    # using model from
    # https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    # but added a rnn for time
    model = Sequential()
    # tensorflow uses w, h, c
    model.add(Permute((2, 3, 1), input_shape=shape,
                      name='permute_input_layer'))
    model.add(Conv2D(16, (4, 4),
                     strides=4,
                     padding='same',
                     activation='relu',
                     name='conv0_open_layer'))
    model.add(MaxPooling2D())
    model.add(Conv2D(4, (2, 2), padding='same',
                     strides=2,
                     activation='relu',
                     name='flattenner_conv1'))
    model.add(MaxPooling2D())
    model.add(Conv2D(1, (1, 1), padding='same',
                     activation='relu',
                     name='conv2'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(env.action_space.n,
                    name='dense1_final_dense'))
    # using sigmoid as sugested by the professor
    model.add(Activation('sigmoid', name='output_layer'))
    print(model.summary())

    # lr value from
    # https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    model.compile(loss='mse', optimizer=Adam(lr=.0025))

    memory = SequentialMemory(limit=10000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05,
                                  nb_steps=100000)
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy,
                   memory=memory, processor=processor, nb_steps_warmup=50,
                   gamma=.99, target_model_update=10, train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.0025), metrics=['mae'])

    data_dir = 'data/'
    weights_filename = data_dir + 'dqn_{}_weights.h5f'.format(ROM)
    if args.mode == 'train':
        # Okay, now it's time to learn something! We capture the interrupt
        # exception so that training can be prematurely aborted. Notice that
        # you can the built-in Keras callbacks!
        checkpoint_weights_filename = (data_dir + 'dqn_' + ROM
                                       + '_weights_{step}.h5f')
        log_filename = data_dir + 'dqn_{}_log.json'.format(ROM)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                             interval=2500)]
        callbacks += [FileLogger(log_filename, interval=100)]
        dqn.fit(env, callbacks=callbacks, nb_steps=args.steps,
                log_interval=10000)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=3, visualize=True)
        print('done')
    else:
        # env.reset()
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)
