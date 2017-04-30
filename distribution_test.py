import numpy as np
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn

def train(f, h, steps):
    batch_size = 32
    xs = h.sample_xs(batch_size)
    ys = h.eval(xs)
    gs = f.eval_grad(ys)
    h.update(xs, gs)

def main():
    x_min = -1
    x_max = 1
    y_min = 0
    y_max = 3
    dx = 0.01
    num_samples = 256
    num_bins = 20
    iter_steps = 1

    f = df.DensityFunc()
    xs = np.arange(-1, 1, 0.01)
    ys = [f.eval(x) for x in xs]

    h = dn.DensityNet()

    i = 0
    while(True):
        train(f, h, iter_steps)

        samples = h.sample(num_samples)
        samples_flat = np.concatenate(samples)

        plt.clf()
        plt.plot(xs, ys, 'g-', label='f(x)')
        plt.hist(samples_flat, num_bins, [x_min, x_max], normed=True, label='samples')

        axes = plt.gca()
        axes.set_xlim([x_min,x_max])
        axes.set_ylim([y_min,y_max])

        plt.title('Iteration: ' + str(i))
        plt.legend()
        plt.xlabel('x')
        plt.pause(0.01)

        i += 1

if __name__ == "__main__":
    main()