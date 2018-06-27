import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.slim as slim

from world import Actions

BATCH_INDEX_MAP = 0
BATCH_INDEX_COORD = 1
BATCH_INDEX_ORIENT = 2
BATCH_INDEX_DEST = 3
BATCH_INDEX_LOAD = 4
BATCH_INDEX_ACTION = 5
BATCH_INDEX_SUCCESS_FLAG = 6
BATCH_INDEX_REWARD = 7


class ExperienceBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.cnt = 0

    def add_to_episode(self, measurement, new_episode=False):
        if len(self.buffer) == 0 or new_episode:
            self.buffer.append([])
        # TODO max length of episode?!
        # if len(self.buffer[-1]) + 1 >= self.buffer_size:
        #     self.buffer[-1][0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer[-1].append(measurement)

    def add(self, episode):
        # episode = np.array([[n for n in m] for m in episode])
        # episode = list(zip(episode))
        self.cnt += 1
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(episode)

    def sample(self, batch_size, trace_length, random_sampler):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            if len(episode) < trace_length:
                ep = (episode[:])
                while len(ep) < trace_length:
                    ep.append(random_sampler())
                sampled_traces.append(ep)
            else:
                point = np.random.randint(0, max(len(episode) + 1 - trace_length, 1))
                sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, -1])


class DoubleQN:
    """Implementation of double Q-Network
    """
    ACTION_SPACE = max(map(int, Actions))+1

    CONV_LAYER1 = (6 ** 2)
    CONV_LAYER2 = (5 ** 2)
    RNN_HIDDEN_LAYER = 3
    DENSE_LAYER1 = 100
    DENSE_LAYER2 = 100
    DENSE_LAYER3 = 20

    @staticmethod
    def random_sampler():
        return [
            np.ones((7, 7)),  # BATCH_INDEX_MAP
            np.random.rand(1, 2),  # BATCH_INDEX_COORD
            np.random.rand(1, 2),  # BATCH_INDEX_ORIEN
            np.random.rand(1, 2),  # BATCH_INDEX_DEST
            np.random.rand(1),  # BATCH_INDEX_LOAD
            np.random.rand(1),  # BATCH_INDEX_ACTION
            np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
            -np.ones(1),  # BATCH_INDEX_REWARD
        ]

    class QNStateActionRNN:
        def __init__(self, horizon=2):
            self.horizon = horizon
            # parameters of the input
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")
            self.trace_length = tf.placeholder(dtype=tf.int32, shape=[], name="trace_length")

            self.combine_state_action()
            self.conv1 = slim.convolution2d(inputs=self.map, num_outputs=DoubleQN.CONV_LAYER1, kernel_size=2, stride=1,
                                            padding='VALID', biases_initializer=None)
            self.conv2 = slim.convolution2d(inputs=self.conv1, num_outputs=DoubleQN.CONV_LAYER2, kernel_size=2, stride=1,
                                            padding='VALID', biases_initializer=None)
            self.combined_input = tf.concat(
                [tf.reshape(self.conv2, shape=[-1,self.conv2.shape[1]*self.conv2.shape[2]]),
                 self.coord,
                 self.orient,
                 self.dest,
                 self.loaded], axis=1)
            self.combined_input = tf.reshape(self.combined_input,
                                             shape=[-1, self.trace_length, self.combined_input.shape[1]])

            rnn_cell = tf.contrib.rnn.BasicLSTMCell(DoubleQN.RNN_HIDDEN_LAYER)
            self.rnn_state_in = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
                inputs=self.combined_input,
                cell=rnn_cell,
                dtype=tf.float32, initial_state=self.rnn_state_in)

            # simple three layer of dence connections
            self.d1 = tf.layers.dense(
                tf.reshape(self.rnn, shape=(-1, DoubleQN.RNN_HIDDEN_LAYER)), DoubleQN.DENSE_LAYER1,
                activation=None)
            self.d2 = tf.layers.dense(self.d1, DoubleQN.DENSE_LAYER2, activation=None)
            self.d3 = tf.layers.dense(self.d2, DoubleQN.DENSE_LAYER3, activation=None)

            self.streamA, self.streamV = tf.split(self.d3, 2, 1)
            self.AW = tf.Variable(tf.random_normal([DoubleQN.DENSE_LAYER3 // 2, DoubleQN.ACTION_SPACE]))
            self.VW = tf.Variable(tf.random_normal([DoubleQN.DENSE_LAYER3 // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            self.q_out = self.Value + tf.subtract(self.Advantage,
                                                  tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

            # self.q_out = tf.layers.dense(self.d3, DoubleQN.ACTION_SPACE)
            self.predict = tf.argmax(self.q_out, 1)

            self.q_out_filt = tf.reduce_sum(tf.multiply(self.q_out, self.action), axis=1)

            self.q_target = tf.placeholder(shape=[None], dtype=tf.float32, name="q_target")

            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q_out_filt))
            trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.update_model = trainer.minimize(self.loss)

        def combine_state_action(self):
            # state
            map_height = 2 * (self.horizon + 1) + 1
            self.map = tf.placeholder(tf.float32, [None, map_height, map_height], name='local_map')
            self.coord = tf.placeholder(tf.float32, [None, 2], name='glob_coord')
            self.orient = tf.placeholder(tf.float32, [None, 2], name='orient')
            self.loaded = tf.placeholder(tf.float32, [None, 1], name='loaded')
            self.dest = tf.placeholder(tf.float32, [None, 2], name='destination')
            # action
            self.action_scalar = tf.placeholder(tf.int32, [None, 1], name='action_scalar')
            self.action = tf.reshape(tf.one_hot(self.action_scalar, DoubleQN.ACTION_SPACE, dtype=tf.float32),
                                     shape=[-1, DoubleQN.ACTION_SPACE])

            # self.combined_input = tf.concat(
            #     [tf.reshape(self.map, shape=[-1, map_height ** 2]),
            #      self.coord,
            #      self.orient,
            #      self.dest,
            #      self.loaded], axis=1)
            # self.combined_input = tf.reshape(self.combined_input,
            #                                  shape=[-1, self.trace_length, self.combined_input.shape[1]])

    def __init__(self, horizon=2, tau=0.001):
        with tf.variable_scope("prime"):
            self.primeQN = DoubleQN.QNStateActionRNN(horizon)
        with tf.variable_scope("secondary"):
            self.secondaryQN = DoubleQN.QNStateActionRNN(horizon)

        prime_vars = tf.trainable_variables(scope="prime")
        second_vars = tf.trainable_variables(scope="secondary")
        self.secondary_updater = \
            self._update_secondary_graph(prime_vars, second_vars, tau)

        self._rnn_state_predict = DoubleQN.get_new_rnn_state()

    @staticmethod
    def get_new_rnn_state():
        return (np.zeros([1, DoubleQN.RNN_HIDDEN_LAYER]),
                np.zeros([1, DoubleQN.RNN_HIDDEN_LAYER]))

    def get_rnn_state(self):
        return self._rnn_state_predict

    def set_rnn_state(self, rnn_state):
        self._rnn_state_predict = rnn_state

    @staticmethod
    def _update_secondary_graph(prime_set, secondary_set, tau):
        assert (len(prime_set) == len(secondary_set))
        op_holder = []
        for idx, var in enumerate(prime_set):
            s_var = secondary_set[idx]
            op_holder.append(s_var.assign((var.value() * tau) + ((1 - tau) * s_var.value())))
        return op_holder

    def update_secondary_network(self, session):
        session.run(self.secondary_updater)

    def predict(self, session, state, epsilon=0.01):
        # facilitate exploration or additional states

        feed_dict = self.fill_feed_dict(state)

        if np.random.rand(1) < epsilon:
            # update state only
            state1, qout = session.run([self.primeQN.rnn_state, self.primeQN.q_out],
                                 feed_dict=feed_dict)
            a = np.random.randint(0, DoubleQN.ACTION_SPACE)
        else:
            a, state1, qout = session.run([self.primeQN.predict, self.primeQN.rnn_state, self.primeQN.q_out],
                                       feed_dict=feed_dict)
            a = a[0]
        self._rnn_state_predict = state1
        return a, state1, qout

    def fill_feed_dict(self, state):
        feed_dict = {
            self.primeQN.batch_size: 1,
            self.primeQN.trace_length: 1,
            self.primeQN.map: state['map'],
            self.primeQN.coord: state['coord'],
            self.primeQN.orient: state['orient'],
            self.primeQN.dest: state['dest'],
            self.primeQN.loaded: state['loaded'],
            # self.primeQN.action_scalar: action,
            self.primeQN.rnn_state_in: self._rnn_state_predict
        }
        return feed_dict

    def train(self, sess, experience: ExperienceBuffer, batch_size, trace_length, y=0.99):
        self.update_secondary_network(sess)

        # Reset the recurrent layer's hidden state
        state_train = (np.zeros([batch_size, DoubleQN.RNN_HIDDEN_LAYER]),
                       np.zeros([batch_size, DoubleQN.RNN_HIDDEN_LAYER]))

        train_batch = experience.sample(batch_size, trace_length, DoubleQN.random_sampler)
        feed_dict = self.train_feed_dict(self.primeQN,
                                         batch_size,
                                         trace_length,
                                         train_batch,
                                         state_train)

        # Below we perform the Double-DQN update to the target Q-values
        Q1 = sess.run(self.primeQN.predict, feed_dict=feed_dict)
        Q2 = sess.run(self.secondaryQN.q_out,
                      feed_dict=self.train_feed_dict(self.secondaryQN,
                                                     batch_size,
                                                     trace_length,
                                                     train_batch,
                                                     state_train))
        end_multiplier = -(train_batch[:, BATCH_INDEX_SUCCESS_FLAG] - 1)

        doubleQ = Q2[range(batch_size * trace_length), Q1]
        # add to reward current predictions
        q_target = train_batch[:, BATCH_INDEX_REWARD] + (y * doubleQ * end_multiplier)

        # Update the network with our target values
        feed_dict[self.primeQN.q_target] = q_target
        loss, _ = sess.run([self.primeQN.loss, self.primeQN.update_model], feed_dict=feed_dict)
        return loss

    def train_feed_dict(self, tn, batch_size, trace_length, train_batch, state_train):
        feed_dict = {
            tn.batch_size: batch_size,
            tn.trace_length: trace_length,
            tn.map: np.stack(train_batch[:, BATCH_INDEX_MAP]),
            tn.coord: np.vstack(train_batch[:, BATCH_INDEX_COORD]),
            tn.orient: np.vstack(train_batch[:, BATCH_INDEX_ORIENT]),
            tn.dest: np.vstack(train_batch[:, BATCH_INDEX_DEST]),
            tn.loaded: np.vstack(train_batch[:, BATCH_INDEX_LOAD]),
            tn.action_scalar: np.stack(train_batch[:, BATCH_INDEX_ACTION]),
            tn.rnn_state_in: state_train
        }
        return feed_dict



if __name__ == '__main__':
    path = "./roboecosys"  # The path to save/load our model to/from.
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print(config.gpu_options)
    # with tf.device('/gpu:0'):
    qb = DoubleQN()
    init = tf.global_variables_initializer()
    #saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=config) as sess:
        # initialize
        #ckpt = tf.train.get_checkpoint_state(path)
        #if ckpt is not None:
        #    print("Loading model")
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        #else:
        #    sess.run(init)
        sess.run(init)
        qb.update_secondary_network(sess)

        experience = ExperienceBuffer()

        # experience.add([
        #     [
        #         np.zeros((7, 7)),  # BATCH_INDEX_MAP
        #         np.zeros((1, 2)),  # BATCH_INDEX_COORD
        #         np.zeros((1, 2)),  # BATCH_INDEX_ORIEN
        #         np.zeros((1, 2)),  # BATCH_INDEX_DEST
        #         np.zeros(1),  # BATCH_INDEX_LOAD
        #         np.zeros(1),  # BATCH_INDEX_ACTION
        #         np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
        #         np.ones(1),  # BATCH_INDEX_REWARD
        #     ],
        #     [
        #         np.ones((7, 7)),  # BATCH_INDEX_MAP
        #         np.ones((1, 2)),  # BATCH_INDEX_COORD
        #         np.ones((1, 2)),  # BATCH_INDEX_ORIEN
        #         np.ones((1, 2)),  # BATCH_INDEX_DEST
        #         np.ones(1),  # BATCH_INDEX_LOAD
        #         np.ones(1),  # BATCH_INDEX_ACTION
        #         np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
        #         np.ones(1),  # BATCH_INDEX_REWARD
        #     ]
        # ])

        experience.add([
            [
                np.random.rand(7, 7),  # BATCH_INDEX_MAP
                np.random.rand(1, 2),  # BATCH_INDEX_COORD
                np.zeros((1, 2)),  # BATCH_INDEX_ORIEN
                np.zeros((1, 2)),  # BATCH_INDEX_DEST
                np.zeros(1),  # BATCH_INDEX_LOAD
                np.zeros(1),  # BATCH_INDEX_ACTION
                np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
                np.ones(1),  # BATCH_INDEX_REWARD
            ]])
        experience.add([
            [
                np.ones((7, 7)),  # BATCH_INDEX_MAP
                np.ones((1, 2)),  # BATCH_INDEX_COORD
                np.ones((1, 2)),  # BATCH_INDEX_ORIEN
                np.ones((1, 2)),  # BATCH_INDEX_DEST
                np.ones(1),  # BATCH_INDEX_LOAD
                np.ones(1)*3,  # BATCH_INDEX_ACTION
                np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
                -np.ones(1),  # BATCH_INDEX_REWARD
            ]
        ])

        experience.add([
            [
                np.ones((7, 7)),  # BATCH_INDEX_MAP
                np.ones((1, 2)),  # BATCH_INDEX_COORD
                np.ones((1, 2)),  # BATCH_INDEX_ORIEN
                np.ones((1, 2)),  # BATCH_INDEX_DEST
                np.ones(1),  # BATCH_INDEX_LOAD
                np.ones(1)*2,  # BATCH_INDEX_ACTION
                np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
                -np.ones(1),  # BATCH_INDEX_REWARD
            ]
        ])


        experience.add([
            [
                np.ones((7, 7)),  # BATCH_INDEX_MAP
                np.ones((1, 2)),  # BATCH_INDEX_COORD
                np.ones((1, 2)),  # BATCH_INDEX_ORIEN
                np.ones((1, 2)),  # BATCH_INDEX_DEST
                np.ones(1),  # BATCH_INDEX_LOAD
                np.ones(1)*3,  # BATCH_INDEX_ACTION
                np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
                -np.ones(1),  # BATCH_INDEX_REWARD
            ]
        ])

        experience.add([
            [
                np.ones((7, 7)),  # BATCH_INDEX_MAP
                np.ones((1, 2)),  # BATCH_INDEX_COORD
                np.ones((1, 2)),  # BATCH_INDEX_ORIEN
                np.ones((1, 2)),  # BATCH_INDEX_DEST
                np.ones(1),  # BATCH_INDEX_LOAD
                np.ones(1)*4,  # BATCH_INDEX_ACTION
                np.ones(1),  # BATCH_INDEX_SUCCESS_FLAG
                -np.ones(1),  # BATCH_INDEX_REWARD
            ]
        ])
        for i in range(1000):
            print(qb.train(sess, experience, 1, 2))

        action, _, q = qb.predict(sess,
                               {
                                   'map': np.zeros((1, 7, 7)),
                                   'coord': np.zeros((1, 2)),
                                   'orient': np.zeros((1, 2)),
                                   'dest': np.zeros((1, 2)),
                                   'loaded': np.zeros((1, 1)),
                               })
        print(action,q)

        action, _, q = qb.predict(sess,
                               {
                                   'map': np.ones((1, 7, 7)),
                                   'coord': np.ones((1, 2)),
                                   'orient': np.ones((1, 2)),
                                   'dest': np.ones((1, 2)),
                                   'loaded': np.ones((1, 1)),
                               })
        print(action,q)

        action, _, q = qb.predict(sess,
                               {
                                   'map': np.ones((1, 7, 7)) * 0.1,
                                   'coord': np.zeros((1, 2)),
                                   'orient': np.ones((1, 2)),
                                   'dest': np.ones((1, 2)),
                                   'loaded': np.ones((1, 1)),
                               })
        print(action,q)

        #saver.save(sess, path + '/model-' + str(1) + '.cptk')
