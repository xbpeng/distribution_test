import numpy as np
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn

def train(f, h):
    return

def main():
    x_min = -1
    x_max = 1
    dx = 0.01
    num_samples = 1000
    num_bins = 20

    f = df.DensityFunc()
    xs = np.arange(-1, 1, 0.01)
    ys = [f.eval(x) for x in xs]

    h = dn.DensityNet()
    train(f, h)

    samples = [h.sample() for i in range(num_samples)]

    plt.plot(xs, ys)
    plt.xlabel('x')
    plt.hist(samples, num_bins, [x_min, x_max], normed=True)
    plt.ylabel('f(x)')
    plt.show()

if __name__ == "__main__":
    main()