import torch

# try to make the distance function more general (only between two vectors and the expand it?)
def compute_lp_dist(x1, x2, p):
    """compute an (m, n) Lp distance matrix from (m, d) and (n, d) matrices 
    with all combination of distances between m and n"""
    return torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=p).squeeze(0)

def compute_rms_dist(x1, x2):
    l2_dist = compute_lp_dist(x1, x2, 2)
    return (l2_dist.pow(2) / x1.size(1)).pow(0.5)