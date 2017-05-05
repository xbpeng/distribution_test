import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import density_func as df
import density_net as dn
import svgd

def train(f, h, steps):
    batch_size = 32
    #entropy_w = 0.2
    entropy_w = 25
    num_samples = 32

    for j in range(steps):
        svgd.step(f, h, batch_size=batch_size, entropy_w = entropy_w, num_samples=num_samples)

def build_density_func():

    d = 2
    if (d == 0):
        means = [np.array([-0.75, 0.4])]
        axes = [np.array([[0.7, 0], 
                          [0, 0.4]])]
        weights = [1]

        theta0 = np.pi * 0.3
        axes[0] = np.array([[np.cos(theta0), -np.sin(theta0)], 
                            [np.sin(theta0), np.cos(theta0)]]).dot(axes[0])
    elif (d == 1):
        means = [np.array([-0.75, 0.5]),
                 np.array([0.75, -0.5])]
        axes = [np.array([[0.7, 0], 
                          [0, 0.3]]),
                np.array([[0.7, 0], 
                          [0, 0.3]])]
        weights = [1, 1]

        theta0 = np.pi * 0.3
        axes[0] = np.array([[np.cos(theta0), -np.sin(theta0)], 
                            [np.sin(theta0), np.cos(theta0)]]).dot(axes[0])

    elif (d == 2):
        means = [np.array([-0.85, 0.5]),
                 np.array([0.85, -0.6]),
                 np.array([0.85, 0.6])]
        axes = [np.array([[0.7, 0], 
                          [0, 0.3]]),
                np.array([[0.5, 0], 
                          [0, 0.2]]),
                np.array([[0.5, 0], 
                          [0, 0.2]])]
        weights = [1, 0.5, 0.5]

        theta0 = np.pi * 0.3
        axes[0] = np.array([[np.cos(theta0), -np.sin(theta0)], 
                            [np.sin(theta0), np.cos(theta0)]]).dot(axes[0])

        theta1 = np.pi * 0.1
        axes[1] = np.array([[np.cos(theta1), -np.sin(theta1)], 
                            [np.sin(theta1), np.cos(theta1)]]).dot(axes[1])

        theta2 = np.pi * -0.2
        axes[2] = np.array([[np.cos(theta2), -np.sin(theta2)], 
                            [np.sin(theta2), np.cos(theta2)]]).dot(axes[2])
    
    covs = [np.outer(A[:,0], A[:,0]) + np.outer(A[:,1], A[:,1]) for A in axes]

    
    f = df.DensityFunc(means, covs, weights)

    return f

def build_net():
    input_dim = 2
    output_dim = 2
    step_size = 0.0002
    
    h = dn.DensityNet(input_dim, output_dim, step_size)
    return h

def color_samples(samples, color_code):
    num_samples = samples.shape[0]
    cols = []
    for i in range(num_samples):
        if (color_code):
            curr_sample = samples[i,:]

            r = (curr_sample[0] + 2) * 0.2
            g = 0
            b = (curr_sample[1] + 2) * 0.2

            r = np.maximum(0, np.minimum(r, 1))
            g = np.maximum(0, np.minimum(g, 1))
            b = np.maximum(0, np.minimum(b, 1))
        else:
            r = 0
            g = 0
            b = 1

        c = [r, g, b]
        cols.append(c)

    return cols

def plot_results(contour_X0, contour_X1, contour_Z, x_min, x_max, dx,
                 sample_xs, samples, mean_pt, iter):
    output_plots = False
    enable_subplots = False
    color_code = True
    draw_correspondence = False

    fig = plt.gcf()
    plt.clf()
        
    if (enable_subplots):
        plt.subplot(1, 2, 2)
        
    CS = plt.contour(contour_X0, contour_X1, contour_Z)
    cols = color_samples(sample_xs, color_code)
    plt.clabel(CS, inline=1, fontsize=10, label='f(x)')
    plt.scatter(samples[:,0], samples[:,1], c=cols, label='samples', s=4, alpha=0.5)
    plt.scatter(mean_pt[0,0], mean_pt[0,1], c='g', label='mean', s=80, marker='P')

    axis0 = plt.gca()
    axis0.set_xlim([x_min, x_max])
    axis0.set_ylim([x_min, x_max])

    plt.title('Iteration: ' + str(iter))
    plt.legend()
    axis0.set_aspect('equal', adjustable='box')

    if (enable_subplots):
        plt.subplot(1, 2, 1)
        plt.scatter(sample_xs[:,0], sample_xs[:,1], c=cols, label='samples', s=4, alpha=0.5)
        axis1 = plt.gca()
        axis1.set_xlim([-3, 3])
        axis1.set_ylim([-3, 3])
        axis1.set_aspect('equal', adjustable='box')

        if (draw_correspondence):
            for p in range(0, samples.shape[0], 20):
                con = matplotlib.patches.ConnectionPatch(xyA=sample_xs[p,:], xyB=samples[p,:], coordsA="data", coordsB="data",
                                                        axesA=axis1, axesB=axis0, alpha=0.5)
                axis1.add_artist(con)
        
    plt.pause(0.01)

    if (output_plots):
        plt.savefig('output/' + str(i).zfill(6) + '.png')

def main():
    x_min = -2
    x_max = 2
    dx = 0.01

    num_samples = 2000
    num_bins = 50
    iter_steps = 50
    
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
    x_size = sample_xs.shape[1]

    while(True):
        train(f, h, iter_steps)
        i += iter_steps

        sample_xs = h.sample_xs(num_samples)
        samples = h.eval(sample_xs)
        mean_pt = h.eval(np.zeros([1, x_size]))
        plot_results(X0, X1, Z, x_min, x_max, dx, sample_xs, samples, mean_pt, i)



if __name__ == "__main__":
    main()