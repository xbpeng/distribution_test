import numpy as np

class DensityFunc(object):

    def __init__(self):
        #self.means = [-0.9, 0.1, 1.1]
        #self.stds = [0.25, 0.2, 0.2]
        #self.weights = [2, 1, 1]
        self.means = [-0.7, 0.7]
        self.stds = [0.3, 0.3]
        self.weights = [1, 1]

        self.weights /= np.sum(self.weights)

        assert(len(self.means) == len(self.stds))
        assert(len(self.means) == len(self.weights))

        return

    def eval(self, x):
        y = 0
        n = len(self.means)
        for i in range(n):
            curr_mean = self.means[i]
            curr_std = self.stds[i]
            curr_weight = self.weights[i]

            curr_y = self.eval_gaussian(x, curr_mean, curr_std)
            y += curr_weight * curr_y

        return y

    def eval_grad_logp(self, x):
        g = 0
        n = len(self.means)

        sum_p = 0
        for i in range(n):
            curr_mean = self.means[i]
            curr_std = self.stds[i]
            curr_weight = self.weights[i]

            curr_p = self.eval_gaussian(x, curr_mean, curr_std)

            delta = x - curr_mean
            curr_g = -0.5 * delta / (curr_std * curr_std)
            g += curr_weight * curr_p * curr_g
            sum_p += curr_weight * curr_p

        g /= sum_p
        return g

    def eval_gaussian(self, x, mean, std):
        delta = x - mean
        p = np.exp(-0.5 * delta * delta / (std * std))
        p /= np.sqrt(2 * np.pi) * std
        return p