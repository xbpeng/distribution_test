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

        med = np.median(dists)
        k = np.exp(-dists / med)
        k *= 2 / (num_samples * med)

        dy = np.transpose(deltas).dot(k)
        gs[i,:] += entropy_w * dy

        # hack
        #sample_gs = f.eval_grad_logp(sample_ys)
        #gs[i,:] += np.transpose(sample_gs).dot(k)
    
    h.update(xs, gs)