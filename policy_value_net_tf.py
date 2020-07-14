# D:\anaconda\envs\tensorflow\python
# _*_ coding:utf-8 _*_
"""
# Time: 2019/2/28  21:53
# Author: AL_Lein
"""
import numpy as np
import tensorflow as tf

from config import *


class PolicyValueNet:
    """ policy-value-network """

    def __init__(self, board_size):
        self.board_size = board_size

        # input & label
        self.input_state = tf.placeholder(tf.float32, shape=[None, 4, board_size, board_size])
        self.value_label = tf.placeholder(tf.float32, shape=[None, 1])
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, board_size ** 2])

        # network
        self.action_probs, self.value_pred = self._build_network()

        # loss
        self.value_loss = tf.losses.mean_squared_error(self.value_label, self.value_pred)
        self.policy_loss = tf.negative(
            tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_probs), 1)))
        self.l2_penalty = L2_WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name.lower()])
        self.loss = self.value_loss + self.policy_loss + self.l2_penalty

        # optimizer & saver
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

        # session & init
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.session.run(init)

    def _build_network(self):
        # 2.commom network layers
        # 2.1 First Convolutional Layer with 32 filters
        x = tf.layers.conv2d(inputs=self.input_state, filters=32, kernel_size=[3, 3], strides=(1, 1), padding='same',
                             data_format='channels_first', activation=None)
        x = tf.layers.batch_normalization(inputs=x)
        x = tf.nn.relu(x)
        # 2.2 residual blocks
        for _ in range(RES_BLOCK_NUM):
            x = self._residual_block(x)

        # 3.Policy Head for generating prior probability vector for each action
        policy = tf.layers.conv2d(inputs=x, filters=2, kernel_size=[1, 1], padding='same', data_format='channels_first',
                                  activation=tf.nn.relu)
        policy = tf.layers.Flatten()(policy)
        action_prob = tf.layers.dense(inputs=policy, units=self.board_size ** 2, activation=tf.nn.softmax)

        # 4.Value Head for generating value of each action
        value = tf.layers.conv2d(inputs=x, filters=1, kernel_size=[1, 1], strides=(1, 1), padding='same',
                                 data_format='channels_first', activation=tf.nn.relu)
        value = tf.layers.Flatten()(value)
        value = tf.layers.dense(inputs=value, units=32, activation=tf.nn.relu)
        value = tf.layers.dense(inputs=value, units=1, activation=tf.nn.tanh)

        return action_prob, value

    def _residual_block(self, x):
        x_shortcut = x
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=(1, 1), padding='same',
                             data_format='channels_first', activation=tf.nn.relu)
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=(1, 1), padding='same',
                             data_format='channels_first', activation=None)
        x = tf.add(x, x_shortcut)
        x = tf.nn.relu(x)
        return x

    def get_policy_value(self, board_state):
        """
        :param board:
        :return: a list of (action, probabilities) tuples for each available action and the score of the board state
        """
        board_state = np.expand_dims(board_state, 0)
        act_probs, value = self.session.run([self.action_probs, self.value_pred],
                                            feed_dict={self.input_state: board_state})
        return act_probs, value

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, _ = self.session.run([self.loss, self.optimizer],
                                   feed_dict={self.input_state: state_batch,
                                              self.mcts_probs: mcts_probs,
                                              self.value_label: winner_batch,
                                              self.learning_rate: lr})
        return loss


if __name__ == '__main__':
    net = PolicyValueNet(3)
