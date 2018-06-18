import tensorflow as tf
import numpy as np
import random

DENSE_LAYER1 = 100
DENSE_LAYER2 = 90
DENSE_LAYER3 = 40

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

    def add(self, episode):
        # episode = np.array([[n for n in m] for m in episode])
        # episode = list(zip(episode))

        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(episode)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, -1])


class DoubleQN:
    """Implementation of double Q-Network
    """
    HORIZON = 2
    RNN_HIDDEN_LAYER = 100
    ACTION_SPACE = 4

    class QNStateActionRNN:
        def __init__(self, horizon=2):
            self.horizon = horizon
            # parameters of the input
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")
            self.trace_length = tf.placeholder(dtype=tf.int32, shape=[], name="trace_length")

            self.combine_state_action()

            rnn_cell = tf.contrib.rnn.BasicLSTMCell(DoubleQN.RNN_HIDDEN_LAYER)
            self.rnn_state_in = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
                inputs=self.combined_input,
                cell=rnn_cell,
                dtype=tf.float32, initial_state=self.rnn_state_in)

            # simple three layer of dence connections
            self.d1 = tf.layers.dense(
                tf.reshape(self.rnn, shape=(-1, DoubleQN.RNN_HIDDEN_LAYER)), DENSE_LAYER1)
            self.d2 = tf.layers.dense(self.d1, DENSE_LAYER2)
            self.d3 = tf.layers.dense(self.d2, DENSE_LAYER3)

            self.streamA, self.streamV = tf.split(self.d3, 2, 1)
            self.AW = tf.Variable(tf.random_normal([DENSE_LAYER3 // 2, DoubleQN.ACTION_SPACE]))
            self.VW = tf.Variable(tf.random_normal([DENSE_LAYER3 // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            self.q_out = self.Value + tf.subtract(self.Advantage,
                                                  tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
            # self.predict = tf.argmax(self.Qout, 1)

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

            self.combined_input = tf.concat(
                [tf.reshape(self.map, shape=[-1, map_height ** 2]),
                 self.coord,
                 self.orient,
                 self.dest,
                 self.loaded,
                 self.action], axis=1)
            self.combined_input = tf.reshape(self.combined_input,
                                             shape=[-1, self.trace_length, self.combined_input.shape[1]])

    def __init__(self, tau=0.001):
        with tf.variable_scope("prime"):
            self.primeQN = DoubleQN.QNStateActionRNN()
        with tf.variable_scope("secondary"):
            self.secondaryQN = DoubleQN.QNStateActionRNN()

        prime_vars = tf.trainable_variables(scope="prime")
        second_vars = tf.trainable_variables(scope="secondary")
        self.secondary_updater = \
            self._update_secondary_graph(prime_vars, second_vars, tau)

        self.rnn_state_predict = (np.zeros([1, DoubleQN.RNN_HIDDEN_LAYER]),
                                  np.zeros([1, DoubleQN.RNN_HIDDEN_LAYER]))

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

    def predict(self, session, state, action, epsilon=0.01):
        # facilitate exploration or additional states

        feed_dict = self.fill_feed_dict(action, state)

        if np.random.rand(1) < epsilon:
            # update state only
            state1 = session.run(self.primeQN.rnn_state,
                                 feed_dict=feed_dict)
            a = np.random.randint(0, DoubleQN.ACTION_SPACE)
        else:
            a, state1 = session.run([self.primeQN.predict, self.primeQN.rnn_state],
                                    feed_dict=feed_dict)
        self.rnn_state_predict = state1
        return a[0], state1

    def fill_feed_dict(self, action, state):
        # TODO make proper mapping
        feed_dict = {
            self.primeQN.batch_size: 1,
            self.primeQN.trace_length: 1,
            self.primeQN.map: state['map'],
            self.primeQN.coord: state['coord'],
            self.primeQN.orient: state['orient'],
            self.primeQN.dest: state['dest'],
            self.primeQN.loaded: state['loaded'],
            self.primeQN.action_scalar: action,
            self.primeQN.rnn_state_in: self.rnn_state_predict
        }
        return feed_dict

    def train(self, session, experience: ExperienceBuffer, batch_size, trace_length, y=0.99):
        self.update_secondary_network(session)

        # Reset the recurrent layer's hidden state
        state_train = (np.zeros([batch_size, DoubleQN.RNN_HIDDEN_LAYER]),
                       np.zeros([batch_size, DoubleQN.RNN_HIDDEN_LAYER]))

        train_batch = experience.sample(batch_size, trace_length)
        feed_dict = self.method_name(self.primeQN, batch_size, trace_length, train_batch, state_train)

        # Below we perform the Double-DQN update to the target Q-values
        Q1 = sess.run(self.primeQN.predict, feed_dict=feed_dict)
        Q2 = sess.run(self.secondaryQN.q_out,
                      feed_dict=self.method_name(self.secondaryQN, batch_size, trace_length, train_batch, state_train))
        end_multiplier = -(train_batch[:, BATCH_INDEX_SUCCESS_FLAG] - 1)

        doubleQ = Q2[range(batch_size * trace_length), Q1]
        # add to reward current predictions
        q_target = train_batch[:, BATCH_INDEX_REWARD] + (y * doubleQ * end_multiplier)

        # Update the network with our target values
        feed_dict[self.primeQN.q_target] = q_target
        loss, _ = sess.run([self.primeQN.loss, self.primeQN.update_model], feed_dict=feed_dict)
        return loss

    def method_name(self, tn, batch_size, trace_length, train_batch, state_train):
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
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print(config.gpu_options)
    # with tf.device('/gpu:0'):
    qb = DoubleQN()
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        # initialize
        sess.run(init)
        qb.update_secondary_network(sess)

        experience = ExperienceBuffer()

        experience.add([
            [
                np.zeros((7, 7)),  # BATCH_INDEX_MAP
                np.zeros((1, 2)),  # BATCH_INDEX_COORD
                np.zeros((1, 2)),  # BATCH_INDEX_ORIEN
                np.zeros((1, 2)),  # BATCH_INDEX_DEST
                np.zeros(1),  # BATCH_INDEX_LOAD
                np.zeros(1),  # BATCH_INDEX_ACTION
                np.zeros(1),  # BATCH_INDEX_SICCES_FLAG
                np.zeros(1),  # BATCH_INDEX_REWARD
            ],
            [
                np.ones((7, 7)),  # BATCH_INDEX_MAP
                np.ones((1, 2)),  # BATCH_INDEX_COORD
                np.ones((1, 2)),  # BATCH_INDEX_ORIEN
                np.ones((1, 2)),  # BATCH_INDEX_DEST
                np.ones(1),  # BATCH_INDEX_LOAD
                np.ones(1),  # BATCH_INDEX_ACTION
                np.ones(1),  # BATCH_INDEX_SICCES_FLAG
                np.ones(1),  # BATCH_INDEX_REWARD
            ]
        ])
        print(qb.train(sess, experience, 1, 2))

        action, _ = qb.predict(sess,
                               {
                                   'map': np.zeros((1, 7, 7)),
                                   'coord': np.zeros((1, 2)),
                                   'orient': np.zeros((1, 2)),
                                   'dest': np.zeros((1, 2)),
                                   'loaded': np.zeros((1, 1)),
                               }, np.zeros((1, 1)))
        print(action)
