import matplotlib.pyplot as plt
import numpy as np
import torch

# https://github.com/coqui-ai/TTS/blob/0592a5805ca7fb877c8fc8df56a4eacac13f2657/TTS/tts/utils/visual.py#L35
def plot_spectrogram(spectrogram, fig_size):
    '''Plot spectrogram.

    Args:
        spectrogram (np.array): Spectrogram values.

    Returns:
        plt.figure
    '''
    if isinstance(spectrogram, torch.Tensor):
        spectrogram_ = spectrogram.detach().cpu().numpy().squeeze().T
    else:
        spectrogram_ = spectrogram.T
    spectrogram_ = spectrogram_.astype(np.float32) if spectrogram_.dtype == np.float16 else spectrogram_
    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram_, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.close()
    return fig

# https://github.com/coqui-ai/TTS/blob/0592a5805ca7fb877c8fc8df56a4eacac13f2657/TTS/tts/utils/visual.py#L52
def plot_pitch(pitch, spectrogram, fig_size):
    '''Plot pitch curves on top of the spectrogram.

    Args:
        pitch ( (T,)np.array): Pitch values.
        spectrogram ( (C, T) np.array): Spectrogram values.

    Returns:
        plt.figure
    '''

    if isinstance(spectrogram, torch.Tensor):
        spectrogram_ = spectrogram.detach().cpu().numpy().squeeze().T
    else:
        spectrogram_ = spectrogram.T
    spectrogram_ = spectrogram_.astype(np.float32) if spectrogram_.dtype == np.float16 else spectrogram_

    old_fig_size = plt.rcParams["figure.figsize"]
    if fig_size is not None:
        plt.rcParams["figure.figsize"] = fig_size

    fig, ax = plt.subplots()

    ax.imshow(spectrogram_, aspect="auto", origin="lower")
    ax.set_xlabel("time")
    ax.set_ylabel("spec_freq")

    ax2 = ax.twinx()
    ax2.plot(pitch, linewidth=5.0, color="red")
    ax2.set_ylabel("F0")

    plt.rcParams["figure.figsize"] = old_fig_size
    plt.close()
    return fig


# Testing Plotting DTW Alignment between pitchs/spectrograms
#def plot_aligned_pitch(x, y, path, sample_init=0, sample_end=400): 
#
#    x = x[sample_init:sample_end]
#    y = y[sample_init:sample_end]
#    path = path[sample_init:sample_end]

#    plt.figure(figsize=(10, 8))
#    dis_y = 150
#    dis_x = 0
#    plt.plot(np.arange(x.shape[0]) + dis_x, x + dis_y, "-", c="C3", linewidth = 3)
#    plt.plot(np.arange(y.shape[0]) - dis_x, y - dis_y, "-", c="C0", linewidth = 3)
#    for x_i, y_j in path:
#        plt.plot([x_i + dis_x, y_j - dis_x], [x[x_i] + dis_y, y[y_j] - dis_y], "-", c="C7", linewidth = 0.5)
#    plt.axis("off")
#    plt.savefig("mygraph.png")

# a = []
# b = []
# a.append(torch.Tensor(refs_yin[0][0]).unsqueeze_(1))
# b.append(torch.Tensor(synths_yin[0][0]).unsqueeze_(1))
# x,y,z,k = batch_dynamic_time_warping(a,b, _compute_rms_dist)
# plot_aligned_pitch(np.array(refs_yin[0][0]), np.array(synths_yin[0][0]), k)