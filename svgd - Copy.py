import numpy as np

def eval_kernel_grad_gaussian(x, samples):
    num_samples = samples.shape[0]
    deltas = x - samples
    dists = np.sum(deltas * deltas, axis=1)

    sort_indices = np.argsort(dists)
    mid = int((num_samples - 1) / 2)
    med_idx = sort_indices[mid]
    med = dists[med_idx]

    m = med / np.log(num_samples)
    k = np.exp(-dists / m)
    k *= 2 / m

    dy = np.transpose(deltas).dot(k)

    return dy

def eval_kernel_grad_invsq(x, samples):
    m = 1
    damping = 0.01

    num_samples = samples.shape[0]
    deltas = x - samples
    dists = np.sum(deltas * deltas, axis=1)

    dists_damped = dists + damping
    k = 1 / (dists_damped * dists_damped)
    k *= m / num_samples

    dy = np.transpose(deltas).dot(k)

    return dy


def eval_kernel_grad_invquad(x, samples):
    m = 7

    num_samples = samples.shape[0]
    deltas = x - samples
    dists = np.sum(deltas * deltas, axis=1)

    k = (m * m * dists)
    k = 1 / (1 + k)
    k *= k
    #k *= 2 * m * m
    k /= num_samples

    k *= 10000 # hack

    dy = np.transpose(deltas).dot(k)

    return dy

def step(f, h, batch_size, entropy_w, num_samples):  
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

        #dy = eval_kernel_grad_gaussian(y, sample_ys)
        #dy = eval_kernel_grad_invsq(y, sample_ys)
        dy = eval_kernel_grad_invquad(y, sample_ys)

        gs[i,:] += entropy_w * dy

        # hack
        #sample_gs = f.eval_grad_logp(sample_ys)
        #gs[i,:] += np.transpose(sample_gs).dot(k)
    
    gs /= batch_size
    h.update(xs, gs)