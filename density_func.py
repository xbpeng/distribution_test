import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath('../OpenAI_Gym'))
import learning.tf_util as U

class DensityFunc(object):

    def __init__(self, means, covs, weights):
        self.means = means
        self.weights = weights

        self.weights /= np.sum(self.weights)
        self.inv_covs = [np.linalg.inv(C) for C in covs]
        self.cov_dets = [np.linalg.det(C) for C in covs]
        
        self._check_valid()
        self._build_comp_graph()

        return

    def get_dim(self):
        dim = 0;
        if (self.get_num_gaussians() > 0):
            dim = len(self.means[0])

        return dim

    def get_num_gaussians(self):
        return len(self.means)

    def eval(self, xs):
        y = self._eval(xs)
        return y

    def eval_grad_logp(self, xs):
        gs = self._eval_grad_logp(xs)
        return gs[0]
    
    def _check_valid(self):
        dim = self.get_dim()
        n = self.get_num_gaussians()

        assert(len(self.means) == n)
        assert(len(self.inv_covs) == n)
        assert(len(self.weights) == n)
        assert(len(self.cov_dets) == n)

        for i in range(n):
            curr_dim = self.means[i].shape[0]
            cov_shape = self.inv_covs[i].shape
            assert curr_dim == dim, "dimension mismatch: expecting %i got %i" % (dim, curr_dim)
            assert cov_shape[0] == dim and cov_shape[1] == dim, "dimension mismatch: expecting %ix%i got %ix%i" % (dim, dim, curr_dim, curr_dim)
        
        return

    def _build_comp_graph(self):
        num_means = self.get_num_gaussians()
        dim = self.get_dim()

        mean_n = np.stack(self.means)
        inv_cov_n = np.stack(self.inv_covs)
        means_reshape = np.reshape(mean_n, [-1, 1, dim])
        inv_cov_reshape = np.reshape(inv_cov_n, [-1, dim, dim])

        x_n = U.Input([None, dim], name='x')
        x_reshape = tf.reshape(x_n, [1, -1, dim])
        diff = x_reshape - means_reshape
        diffT = tf.transpose(diff, perm=[0, 2, 1])
        Cd = tf.matmul(tf.cast(inv_cov_reshape, tf.float32), diffT)
        CdT = tf.transpose(Cd, perm=[0, 2, 1])
        dCd = tf.reduce_sum(diff * CdT, axis=2)
        
        w_n = np.array(self.weights)
        w_n = w_n[:,np.newaxis]
        z_n = np.array(self.cov_dets)
        z_n = ((2 * np.pi) ** (0.5 * dim)) * np.sqrt(z_n)

        exp_val = tf.exp(-0.5 * dCd)
        exp_valT = tf.transpose(exp_val)
        exp_valT_z = exp_valT / z_n
        p = tf.matmul(exp_valT_z, tf.cast(w_n, tf.float32))

        logp = tf.log(p)
        grad_logp = tf.gradients(logp, x_n)

        self._eval = U.function([x_n], p)
        self._eval_grad_logp = U.function([x_n], grad_logp)

        return