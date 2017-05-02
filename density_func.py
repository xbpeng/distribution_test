import numpy as np

class DensityFunc(object):

    def __init__(self, means, covs, weights):
        #self.means = [-0.9, 0.1, 1.1]
        #self.covs = [0.25, 0.2, 0.2]
        #self.weights = [2, 1, 1]
        self.means = means
        self.covs = covs
        self.weights = weights

        self.weights /= np.sum(self.weights)

        self._check_valid()
        
        return

    def get_dim(self):
        dim = 0;
        if (self.get_num_gaussians() > 0):
            dim = len(self.means[0])

        return dim

    def get_num_gaussians(self):
        return len(self.means)

    def eval(self, x):
        y = 0
        n = len(self.means)
        for i in range(n):
            curr_mean = self.means[i]
            curr_cov = self.covs[i]
            curr_weight = self.weights[i]

            curr_y = self.eval_gaussian(x, curr_mean, curr_cov)
            y += curr_weight * curr_y

        return y

    def eval_grad_logp(self, x):
        g = 0
        n = len(self.means)

        sum_p = 0
        for i in range(n):
            curr_mean = self.means[i]
            curr_cov = self.covs[i]
            curr_weight = self.weights[i]

            curr_p = self.eval_gaussian(x, curr_mean, curr_cov)

            delta = x - curr_mean
            curr_g = -0.5 * delta / curr_cov[0, 0]
            g += curr_weight * curr_p * curr_g
            sum_p += curr_weight * curr_p

        g /= sum_p
        return g

    def eval_gaussian(self, x, mean, cov):
        delta = x - mean
        p = np.exp(-0.5 * delta * delta / cov[0, 0])
        p /= np.sqrt(2 * np.pi * cov[0, 0])
        return p

    def _check_valid(self):
        dim = self.get_dim()
        n = self.get_num_gaussians()

        assert(len(self.means) == n)
        assert(len(self.means) == n)
        assert(len(self.weights) == n)
       
        for i in range(n):
            curr_dim = self.means[i].shape[0]
            cov_shape = self.covs[i].shape
            assert curr_dim == dim, "dimension mismatch: expecting %i got %i" % (dim, curr_dim)
            assert cov_shape[0] == dim and cov_shape[1] == dim, "dimension mismatch: expecting %ix%i got %ix%i" % (dim, dim, curr_dim, curr_dim)
        
        return