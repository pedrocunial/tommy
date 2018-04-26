### Mostly from A. Géron's book


import gym
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected

CART_ROM = 'CartPole-v0'
N_ITERATIONS = 250
N_MAX_STEPS = 1000
N_GAMES_PER_UPDATE = 10
SAVE_ITERATIONS = 10
DISCOUNT_RATE = 0.95

def basic_policy(obs):
    # obs[2] = angle left
    return 0 if obs[2] < 0 else 1

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

if __name__ == '__main__':
    totals = []
    env = gym.make(CART_ROM)
    N_INPUTS = env.observation_space.shape[0]
    N_HIDDEN = 4
    N_OUTPUTS = 1
    LEARNING_RATE = 0.01
    initializer = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, N_INPUTS])
    hidden = fully_connected(X, N_HIDDEN, activation_fn=tf.nn.elu,
                             weights_initializer=initializer)
    logits = fully_connected(hidden, N_OUTPUTS, activation_fn=None,
                             weights_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)

    prob_left_right = tf.concat(axis=1, values=[outputs, 1-outputs])
    action = tf.multinomial(tf.log(prob_left_right), num_samples=1)

    y = 1. - tf.to_float(action)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    gradients = [g for g, v in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, var in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, var))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for iteration in range(N_ITERATIONS):
            all_rewards = []
            all_gradients = []
            for game in range(N_GAMES_PER_UPDATE):
                current_rewards = []
                current_gradients = []
                obs = env.reset()
                for step in range(N_MAX_STEPS):
                    action_val, gradients_val = sess.run(
                        [action, gradients],
                        feed_dict={X: obs.reshape(1, N_INPUTS)}
                    )
                    obs, reward, done, info = env.step(action_val[0][0])
                    env.render()
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

            # check policy for 10 episodes and then update
            all_rewards = discount_and_normalize_rewards(all_rewards,
                                                         DISCOUNT_RATE)
            feed_dict = {}
            for var_idx, grad_placeholder in enumerate(gradient_placeholders):
                mean_grads = np.mean(
                    [reward * all_gradients[game_idx][step][var_idx]
                     for game_idx, rewards in enumerate(all_rewards)
                     for step, reward in enumerate(rewards)],
                    axis=0)
                feed_dict[grad_placeholder] = mean_grads
            sess.run(training_op, feed_dict=feed_dict)
            if iteration % SAVE_ITERATIONS:
                saver.save(sess, './my_policy_net_pg.cpkt')

    # for episode in range(N_EPISODES):
    #     episode_rewards = 0
    #     obs = env.reset()
    #     for step in range(LEN_EPISODE):  # better than inf loop
    #         action = basic_policy(obs)
    #         obs, reward, done, info = env.step(action)
    #         img = env.render(mode='rgb_array')
    #         episode_rewards += reward
    #         if done:
    #             break
    #     totals.append(episode_rewards)
    # env.close()
    # print('mean: {} | stddev: {} | min: {} | max: {}'.format(
    #     np.mean(totals), np.std(totals), np.min(totals), np.max(totals)
    # ))
