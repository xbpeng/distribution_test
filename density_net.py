import numpy as np
import sys
import os
import tensorflow as tf

sys.path.append(os.path.abspath('../OpenAI_Gym'))
import learning.tf_util as U

class DensityNet(object):

    def __init__(self):
        self.input_dim = 1
        self.output_dim = 1
        step_size = 0.005

        x_n = U.Input([None, self.input_dim], name='x')
        g_n = U.Input([None, self.output_dim], name='g')

        h1 = U.relu(U.dense(x_n, 64, weight_init=U.Xavier(1.0)))
        h2 = U.relu(U.dense(h1, 32, weight_init=U.Xavier(1.0)))
        output = U.dense(h2, self.output_dim, weight_init=U.NormalizedColumns(1.0))

        # hack
        #net_params = tf.trainable_variables()
        #grads = tf.gradients(output, net_params, -g_n)
        #update_op = tf.train.AdamOptimizer(step_size).apply_gradients(zip(grads, net_params))
        
        loss = 0.5 * U.mean(U.square(g_n - output))
        update_op = tf.train.AdamOptimizer(step_size).minimize(loss)

        self.eval = U.function([x_n], output)
        self._step = U.function([x_n, g_n], update_op)

        U.initialize()

        return

    def sample(self):
        return sample(self, 1)[0]

    def sample(self, n):
        xs = self.sample_xs(n)
        y = self.eval(xs)
        return y

    def sample_xs(self, n):
        # hack
        #xs = np.array([np.random.randn(self.input_dim) for i in range(n)])
        xs = np.array([np.random.rand(self.input_dim) * 2 - 1 for i in range(n)])
        return xs

    def update(self, x_n, g_n):
        self._step(x_n, g_n)