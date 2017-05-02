import numpy as np
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn
import svgd

def train(f, h, steps):
    batch_size = 32
    entropy_w = 2
    num_samples = 32

    for j in range(steps):
        svgd.step(f, h, batch_size=batch_size, entropy_w = entropy_w, num_samples=num_samples)

def BuildDensityFunc():
    means = [np.array([-1, 0]), 
             np.array([1, 0])]
    covs = [np.array([[0.5 * 0.5, 0], 
                      [0, 0.3 * 0.3]]), 
            np.array([[0.3 * 0.3, 0], 
                      [0, 0.3 * 0.3]])]

    theta = np.pi * 0.25
    covs[0] = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]]).dot(covs[0])

    weights = [1, 1]
    f = df.DensityFunc(means, covs, weights)

    return f

def main():
    x_min = -3
    x_max = 3
    dx = 0.01

    input_dim = 1
    output_dim = 1
    num_samples = 1000
    num_bins = 50
    iter_steps = 20
    
    f = BuildDensityFunc()
    h = dn.DensityNet(input_dim, output_dim)

    xs = np.arange(x_min, x_max, dx)
    X0, X1 = np.meshgrid(xs, xs)
    X = np.transpose(np.stack([X0.flatten(), X1.flatten()]))

    Z = f.eval(X)
    Z = Z.reshape(X0.shape)
    #gs = f.eval_grad_logp(xs)

    i = 0
    while(True):
        #train(f, h, iter_steps)
        i += iter_steps

        #samples = h.sample(num_samples)
        #samples_flat = np.concatenate(samples)

        plt.clf()
        CS = plt.contour(X0, X1, Z)
        plt.clabel(CS, inline=1, fontsize=10)
        #plt.plot(xs, ys, 'g-', label='f(x)')
        #plt.hist(samples_flat, num_bins, [x_min, x_max], normed=True, label='samples')
        #plt.plot(xs, gs, label='g(x)')

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