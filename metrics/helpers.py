# https://github.com/bastibe/MAPS-Scripts/blob/master/helper.py
# GPL 3.0 LICENSE
import numpy as np
import torch
from metrics.DTW import batch_dynamic_time_warping
from scipy.interpolate import interp1d

def same_t_in_true_and_est(func):
    def new_func(true_t, true_f, est_t, est_f):
        assert type(true_t) is np.ndarray
        assert type(true_f) is np.ndarray
        assert type(est_t) is np.ndarray
        assert type(est_f) is np.ndarray

        interpolated_f = interp1d(
            est_t, est_f, bounds_error=False, kind='nearest', fill_value=0
        )(true_t)
        return func(true_t, true_f, true_t, interpolated_f)
    return new_func

def gross_pitch_error_frames(true_t, true_f, est_t, est_f, eps=1e-8):
    voiced_frames = true_voiced_frames(true_t, true_f, est_t, est_f)
    true_f_p_eps = [x + eps for x in true_f]
    pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
    return voiced_frames & pitch_error_frames

def voicing_decision_error_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) != (true_f != 0)

def true_voiced_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) & (true_f != 0)


def batch_compute_distortion(y1, y2, sr, feat_fn, dist_fn, normalize_align_type):
    # Distances, Sizes, Spectrograms of Ref, Spectrograms of Synths
    s, x1, x2 = [], [], []
    
    # Loop throug each (Ref, Synth) pair
    for cur_y1, cur_y2 in zip(y1, y2):
        assert cur_y1.ndim == 1 and cur_y2.ndim == 1

        # Compute Mel-Specs for MSD or MFCC for MCD
        cur_x1 = feat_fn(cur_y1)
        cur_x2 = feat_fn(cur_y2)
        x1.append(cur_x1)
        x2.append(cur_x2)
        
        # Save Sizes for UnPadding
        size = torch.empty(cur_x1.size(0), cur_x2.size(0))
        s.append(size.size())
    s = torch.LongTensor(s).to(cur_y1.device)
    
    # Get DTW D matrix, Minimal Cost Paths, Optimal Path
    return batch_dynamic_time_warping(x1,x2,dist_fn, normalize_align_type)['norm_align_costs']


def add_basic_stats(dic):
    mean = np.mean(list(dic.values()))
    std = np.std(list(dic.values()))
    max = np.max(list(dic.values()))
    argmax = float(np.argmax(list(dic.values())))
    min = np.min(list(dic.values()))
    argmin = float(np.argmin(list(dic.values())))
    dic['stats'] = {'mean':mean, 'std':std, 'max':max, 'argmax':argmax, 'min':min, 'argmin':argmin}
    return dic