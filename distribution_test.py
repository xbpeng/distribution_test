import numpy as np
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn

def train(f, h, steps):
    batch_size = 32
    entropy_w = 2
    num_samples = 32
    
    for j in range(steps):
        xs = h.sample_xs(batch_size)
        ys = h.eval(xs)
        gs = f.eval_grad_logp(ys)

        # hack
        #gs.fill(0)
        for i in range(batch_size):
            x = xs[i,:]
            y = ys[i,:]
            sample_xs = h.sample_xs(num_samples)
            sample_ys = h.eval(sample_xs)

            deltas = y - sample_ys
            dists = np.sum(deltas * deltas, axis=1)

            med = np.median(dists)
            k = np.exp(-dists / med)
            k *= 2 / (num_samples * med)

            dy = np.transpose(deltas).dot(k)
            gs[i,:] += entropy_w * dy

            # hack
            #sample_gs = f.eval_grad_logp(sample_ys)
            #gs[i,:] += np.transpose(sample_gs).dot(k)
    
        h.update(xs, gs)

def main():
    x_min = -2
    x_max = 2
    y_min = 0
    y_max = 2
    dx = 0.01
    num_samples = 1000
    num_bins = 50
    iter_steps = 20
    input_dim = 1
    output_dim = 1

    f = df.DensityFunc()
    xs = np.arange(x_min, x_max, 0.01)
    ys = f.eval(xs)
    #gs = f.eval_grad_logp(xs)

    h = dn.DensityNet(input_dim, output_dim)

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