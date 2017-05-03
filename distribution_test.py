import numpy as np
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn
import svgd

def train(f, h, steps):
    batch_size = 32
    entropy_w = 5
    num_samples = 32

    for j in range(steps):
        svgd.step(f, h, batch_size=batch_size, entropy_w = entropy_w, num_samples=num_samples)

def build_density_func():
    means = [np.array([-0.7]), 
             np.array([0.7])]
    covs = [np.array([[0.3 * 0.3]]), 
            np.array([[0.3 * 0.3]])]
    weights = [1, 1]
    f = df.DensityFunc(means, covs, weights)

    return f

def build_net():
    input_dim = 1
    output_dim = 1
    step_size = 0.001
    
    h = dn.DensityNet(input_dim, output_dim, step_size)
    return h

def main():
    x_min = -2
    x_max = 2
    y_min = 0
    y_max = 2
    dx = 0.01

    num_samples = 1000
    num_bins = 50
    iter_steps = 5
    
    f = build_density_func()
    h = build_net()

    xs = np.arange(x_min, x_max, dx)
    xs = xs.reshape(-1, 1)
    ys = f.eval(xs)
    #gs = f.eval_grad_logp(xs)

    i = 0
    while(True):
        train(f, h, iter_steps)
        i += iter_steps

        samples = h.sample(num_samples)
        samples_flat = np.concatenate(samples)

        plt.clf()
        plt.plot(xs, ys, 'g-', label='f(x)')
        plt.hist(samples_flat, num_bins, [x_min, x_max], normed=True, label='samples')
        #plt.plot(xs, gs, label='g(x)')

        axes = plt.gca()
        axes.set_xlim([x_min,x_max])
        axes.set_ylim([y_min,y_max])

        plt.title('Iteration: ' + str(i))
        plt.legend()
        plt.xlabel('x')
        plt.pause(0.01)
        #plt.savefig('output/' + str(i).zfill(6) + '.png')


if __name__ == "__main__":
    main()