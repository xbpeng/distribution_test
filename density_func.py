import numpy as np

class DensityFunc(object):

    def __init__(self):
        self.mean = 0
        self.stdev = 0.5

    def eval(self, x):
        delta = x - self.mean
        y = np.exp(-0.5 * delta * delta / (self.stdev * self.stdev))
        y /= np.sqrt(2 * np.pi) * self.stdev
        return y

    def eval_grad(self, x):
        g = self.eval(x)
        g *= -(x - self.mean) / (self.stdev * self.stdev)
        return g