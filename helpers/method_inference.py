import torch
import numpy as np

def forward_haploid(net, x, do_mean=True):
    out_base = net.forward_base(x)
    out_base = torch.stack([out_base, out_base], dim=3)
    out = net.forward_smoother(out_base)[:,:,:,0]
    if do_mean:
        out = torch.mean(out, dim=2)
    pred = torch.argmax(out, dim=1)
    return out, pred


def forward_nadm(nadm, x):
    _, out = nadm(x)
    prob = out[0]
    pred = torch.argmax(prob, dim=1)
    return prob, pred



def extend_if_lai(y, shape):
    if len(shape) > 2:
        y_ = y.unsqueeze(1).repeat(1,shape[3])
    else:
        y_ = y
    return y_


def compute_pca(x, W, b):
    pcs = torch.matmul((x - b),W.T)
    mean_pcs = torch.mean(pcs, dim=0)
    cov_pcs = torch.matmul((pcs-mean_pcs).T,(pcs-mean_pcs))
    std_pcs = torch.std(pcs, dim=0)
    return pcs, mean_pcs, cov_pcs, std_pcs


def unique_to_percentage(unique, count, max_value):
    hist = np.zeros(max_value)
    hist[unique] += count
    hist /= np.sum(hist)
    return hist
