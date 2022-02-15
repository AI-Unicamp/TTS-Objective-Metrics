import argparse
import librosa
import numpy as np
import torch
from config.global_config import GlobalConfig
from audio.pitch import dio, yin
from metrics.dists import compute_rms_dist

# https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/tasks/text_to_speech.py#L304
def batch_dynamic_time_warping(x1, x2, dist_fn, norm_align_type):
    """full batched DTW without any constraints
    x1:  list of tensors of the references
    x2:  list of tensors of the synthesizeds
    dist_fn: distance metric used to compute
    outputs: list of floats/ints or list of numpy arrays indexed by batch
    """

    distance, shapes = [], []
    for cur_x1, cur_x2 in zip(x1, x2):
        # Compute Distance between the Ref/Synths
        cur_d = eval(dist_fn)(cur_x1, cur_x2)
        # cur_d = dist_fn(cur_x1, cur_x2)
        distance.append(cur_d)
        shapes.append(distance[-1].size())

    # Get Max Spec Size of References
    max_m = max(ss[0] for ss in shapes)
    # Get Max Spec Size of Synths
    max_n = max(ss[1] for ss in shapes)
    
    # Pad all distance matrices with to be [Max_Synth, Max_Ref]
    distance = torch.stack(
        [torch.nn.functional.pad(dd, (0, max_n - dd.size(1), 0, max_m - dd.size(0))) for dd in distance]
    )
    
    # All Spec Sizes (List of [REF_SPEC_SIZE, SYNTH_SPEC_SIZE])
    shapes = torch.LongTensor(shapes).to(distance.device)
    
    # ptr: 0=left, 1=up-left, 2=up
    ptr2dij = {0: (0, -1), 1: (-1, -1), 2: (-1, 0)}

    bsz, m, n = distance.size()
    cumdist = torch.zeros_like(distance)
    backptr = torch.zeros_like(distance).type(torch.int32) - 1

    # initialize
    cumdist[:, 0, :] = distance[:, 0, :].cumsum(dim=-1)
    cumdist[:, :, 0] = distance[:, :, 0].cumsum(dim=-1)
    backptr[:, 0, :] = 0
    backptr[:, :, 0] = 2

    # DP with optimized anti-diagonal parallelization, O(M+N) steps
    for offset in range(2, m + n - 1):
        ind = _antidiag_indices(offset, 1, m, 1, n)
        c = torch.stack(
            [
                cumdist[:, ind[0], ind[1] - 1],
                cumdist[:, ind[0] - 1, ind[1] - 1],
                cumdist[:, ind[0] - 1, ind[1]],
            ],
            dim=2,
        )
        v, b = c.min(axis=-1)
        
        # Best Backpaths from each starting point
        backptr[:, ind[0], ind[1]] = b.int()

        # D(i,j) = d(x_i,y_i) + min(D(i-1,j-1), D(i-1,j), D(i,j-1))
        cumdist[:, ind[0], ind[1]] = v + distance[:, ind[0], ind[1]]

    # Evaluate the Optimal Backpath
    pathmap = torch.zeros_like(backptr)
    for b in range(bsz):
        i = m - 1 if shapes is None else (shapes[b][0] - 1).item()
        j = n - 1 if shapes is None else (shapes[b][1] - 1).item()
        dtwpath = [(i, j)]
        while (i != 0 or j != 0) and len(dtwpath) < 10000:
            assert i >= 0 and j >= 0
            di, dj = ptr2dij[backptr[b, i, j].item()]
            i, j = i + di, j + dj
            dtwpath.append((i, j))
        dtwpath = dtwpath[::-1]
        indices = torch.from_numpy(np.array(dtwpath)).long()
        pathmap[b, indices[:, 0], indices[:, 1]] = 1
     
    norm_align_costs = []
    cumdists = []
    backptrs = []
    pathmaps = []
    idxs = []
    itr = zip(shapes, cumdist, backptr, pathmap)

    # Get distortion for each item in the batch
    for (m, n), cumdist, backptr, pathmap in itr:
        cumdist = cumdist[:m, :n]
        backptr = backptr[:m, :n]
        pathmap = pathmap[:m, :n]
        divisor = _get_divisor(pathmap, norm_align_type)
        norm_align_cost = cumdist[-1, -1] / divisor
        
        # Get Indices
        p = pathmap.squeeze(0).cpu().detach().numpy()
        idx = np.transpose(np.nonzero(p))

        norm_align_costs.append(float(norm_align_cost.item()))
        cumdists.append(cumdist.numpy())
        backptrs.append(backptr.numpy())
        pathmaps.append(pathmap.numpy())
        idxs.append(idx)

    return {'norm_align_costs' : norm_align_costs, 'cumdists' : cumdists, 'backptrs' : backptrs, 'pathmaps' : pathmaps, 'idxs': idxs}

def _antidiag_indices(offset, min_i=0, max_i=None, min_j=0, max_j=None):
    """
    for a (3, 4) matrix with min_i=1, max_i=3, min_j=1, max_j=4, outputs
    offset=2 (1, 1),
    offset=3 (2, 1), (1, 2)
    offset=4 (2, 2), (1, 3)
    offset=5 (2, 3)
    constraints:
        i + j = offset
        min_j <= j < max_j
        min_i <= offset - j < max_i
    """
    if max_i is None:
        max_i = offset + 1
    if max_j is None:
        max_j = offset + 1
    min_j = max(min_j, offset - max_i + 1, 0)
    max_j = min(max_j, offset - min_i + 1, offset + 1)
    j = torch.arange(min_j, max_j)
    i = offset - j
    return torch.stack([i, j])

def _get_divisor(pathmap, normalize_type):
    if normalize_type is None:
        return 1
    elif normalize_type == "len1":
        return pathmap.size(0)
    elif normalize_type == "len2":
        return pathmap.size(1)
    elif normalize_type == "path":
        return pathmap.sum().item()
    else:
        raise ValueError(f"normalize_type {normalize_type} not supported")

def main(gt_path, synth_path, pitch_algorithm):
    
    # Load Audio
    x_gt, _ = librosa.load(gt_path)
    x_synth, _ = librosa.load(synth_path)
    
    # Load Config
    config = GlobalConfig()
    
    # Compute Pitch
    pitch_gt = eval(pitch_algorithm)(x_gt, config)
    pitch_synth = eval(pitch_algorithm)(x_synth, config)

    # Initiate Lists of Tensors for Batched DTW
    x1 = []
    x2 = []

    # Append Each Tensor
    x1.append(torch.Tensor(pitch_gt['pitches']).unsqueeze_(1))
    x2.append(torch.Tensor(pitch_synth['pitches']).unsqueeze_(1))

    return print(batch_dynamic_time_warping(x1, x2, config.dist_fn, config.norm_align_type)['norm_align_costs'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to calculate the Dynamic Time Warping between the ground truth and the synthesized audios.')
    parser.add_argument("--gt_path", required = True, type=str, help = 'Path to corresponding ground truth audio in .wav format')
    parser.add_argument("--synth_path", required = True, type=str, help = 'Path to corresponding synthesized audio in .wav format')
    parser.add_argument("--pitch_algorithm", required = True, type=str, choices = ["dio", "yin"], help = 'Choose method of computing pitch')
    args = parser.parse_args()

    gt_path = args.gt_path
    synth_path = args.synth_path
    pitch_algorithm = args.pitch_algorithm

    main(gt_path, synth_path, pitch_algorithm)