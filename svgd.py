import numpy as np

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

        deltas = y - sample_ys
        dists = np.sum(deltas * deltas, axis=1)

        sort_indices = np.argsort(dists)
        med_idx = sort_indices[int((batch_size - 1) / 2)]
        med = dists[med_idx]

        m = med / np.log(num_samples)
        k = np.exp(-dists / m)
        k *= 2 / m

        dy = np.transpose(deltas).dot(k)
        gs[i,:] += entropy_w * dy

        # hack
        #sample_gs = f.eval_grad_logp(sample_ys)
        #gs[i,:] += np.transpose(sample_gs).dot(k)
    
    gs /= batch_size
    h.update(xs, gs)