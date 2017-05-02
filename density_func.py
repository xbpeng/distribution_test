import numpy as np

class DensityFunc(object):

    def __init__(self, means, covs, weights):
        #self.means = [-0.9, 0.1, 1.1]
        #self.covs = [0.25, 0.2, 0.2]
        #self.weights = [2, 1, 1]
        self.means = means
        self.weights = weights

        self.weights /= np.sum(self.weights)
        self.inv_covs = [np.linalg.inv(C) for C in covs]
        self.cov_dets = [np.linalg.det(C) for C in covs]

        self._check_valid()
        
        return

    def get_dim(self):
        dim = 0;
        if (self.get_num_gaussians() > 0):
            dim = len(self.means[0])

        return dim

    def get_num_gaussians(self):
        return len(self.means)

    def eval(self, xs):
        y = 0
        n = len(self.means)
        for i in range(n):
            curr_mean = self.means[i]
            curr_inv_cov = self.inv_covs[i]
            curr_weight = self.weights[i]
            curr_det = self.cov_dets[i]

            curr_y = self._eval_gaussian(xs, curr_mean, curr_inv_cov, curr_det)
            y += curr_weight * curr_y

        return y

    def eval_grad_logp(self, xs):
        dim = self.get_dim()
        n = len(self.means)
        gs = np.zeros([xs.shape[0], dim])

        sum_ps = np.zeros(xs.shape[0])
        for i in range(n):
            curr_mean = self.means[i]
            curr_inv_cov = self.inv_covs[i]
            curr_weight = self.weights[i]
            curr_det = self.cov_dets[i]

            curr_ps = self._eval_gaussian(xs, curr_mean, curr_inv_cov, curr_det)

            deltas = xs - curr_mean
            curr_gs = -0.5 * deltas.dot(np.transpose(curr_inv_cov))
            gs += curr_weight * (curr_gs * curr_ps[:, np.newaxis])
            sum_ps += curr_weight * curr_ps

        gs = gs / sum_ps[:, np.newaxis]
        return gs

    def _eval_gaussian(self, xs, mean, inv_cov, det):
        delta = xs - mean
        dim = self.get_dim()
        exp_vals = delta.dot(np.transpose(inv_cov))
        exp_vals = np.multiply(delta, exp_vals)
        exp_vals = np.sum(exp_vals, 1)

        p = np.exp(-0.5 * exp_vals)
        p /= ((2 * np.pi) ** (0.5 * dim)) * np.sqrt(det)
        return p

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