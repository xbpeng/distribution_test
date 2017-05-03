import numpy as np
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn
import svgd

def train(f, h, steps):
    batch_size = 32
    entropy_w = 0.1
    num_samples = 32

    for j in range(steps):
        svgd.step(f, h, batch_size=batch_size, entropy_w = entropy_w, num_samples=num_samples)

def build_density_func():
    #means = [np.array([-0.75, 0.4]),
    #         np.array([0.75, -0.4])]
    #axes = [np.array([[0.7, 0], 
    #                  [0, 0.3]]),
    #        np.array([[0.7, 0], 
    #                  [0, 0.3]])]
    #weights = [1, 1]

    means = [np.array([-0.75, 0.4])]
    axes = [np.array([[0.7, 0], 
                      [0, 0.4]])]
    weights = [1]

    theta = np.pi * 0.3
    axes[0] = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]]).dot(axes[0])

    covs = [np.outer(A[:,0], A[:,0]) + np.outer(A[:,1], A[:,1]) for A in axes]

    
    f = df.DensityFunc(means, covs, weights)

    return f

def build_net():
    input_dim = 2
    output_dim = 2
    step_size = 0.001
    
    h = dn.DensityNet(input_dim, output_dim, step_size)
    return h

def main():
    x_min = -2
    x_max = 2
    dx = 0.01

    num_samples = 2000
    num_bins = 50
    iter_steps = 20
    
    f = build_density_func()
    h = build_net()

    xs = np.arange(x_min, x_max, dx)
    X0, X1 = np.meshgrid(xs, xs)
    X = np.transpose(np.stack([X0.flatten(), X1.flatten()]))

    Z = f.eval(X)
    Z = Z.reshape(X0.shape)
    #gs = f.eval_grad_logp(xs)

    i = 0
    sample_xs = h.sample_xs(num_samples)

    while(True):
        train(f, h, iter_steps)
        i += iter_steps

        sample_xs = h.sample_xs(num_samples)
        samples = h.eval(sample_xs)

        plt.clf()
        CS = plt.contour(X0, X1, Z)
        plt.clabel(CS, inline=1, fontsize=10, label='f(x)')
        plt.scatter(samples[:,0], samples[:,1], label='samples', s=4, alpha=0.5)

        axes = plt.gca()
        axes.set_xlim([x_min, x_max])
        axes.set_ylim([x_min, x_max])

        plt.title('Iteration: ' + str(i))
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.01)
        #plt.savefig('output/' + str(i).zfill(6) + '.png')


if __name__ == "__main__":
    main()