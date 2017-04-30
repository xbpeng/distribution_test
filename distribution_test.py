import numpy as np
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn

def train(f, h, steps):
    return

def main():
    x_min = -1
    x_max = 1
    dx = 0.01
    num_samples = 1000
    num_bins = 20
    iter_steps = 1

    f = df.DensityFunc()
    xs = np.arange(-1, 1, 0.01)
    ys = [f.eval(x) for x in xs]

    h = dn.DensityNet()


    train(f, h, iter_steps)
    samples = [h.sample() for i in range(num_samples)]
    
    plt.hist(samples, num_bins, [x_min, x_max], normed=True, label='samples')
    plt.plot(xs, ys, 'g-', label='f(x)')

    plt.legend()
    plt.xlabel('x')
    plt.show()

if __name__ == "__main__":
    main()