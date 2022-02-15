import argparse
from config.global_config import GlobalConfig
import librosa
import torch
import torchaudio
from metrics.helpers import batch_compute_distortion

# https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/speech_synthesis/utils.py
def batch_mel_cepstral_distortion(y1, y2, config):
    """
    https://arxiv.org/pdf/2011.03568.pdf
    The root mean squared error computed on 13-dimensional MFCC using DTW for
    alignment. MFCC features are computed from an 80-channel log-mel
    spectrogram using a 50ms Hann window and hop of 12.5ms.
    y1: list of arrays of waveforms of type double
    y2: list of arrays of waveforms of type double
    sr: sampling rate
    """
    
    return batch_compute_distortion(
        y1,
        y2,
        config.sr,
        lambda y: config.mfcc_fn.to(y1[0].device)(y).transpose(-1, -2),
        'compute_rms_dist',
        config.norm_align_type,
    )

def main(gt_path, synth_path):
    
    # Load Audio
    x_gt, sr_gt = librosa.load(gt_path)
    x_synth, sr_synth = librosa.load(synth_path)

    # Load Config
    config = GlobalConfig()

    # Initiate Lists of Tensors for Batched DTW
    x1 = []
    x2 = []

    # Append Each Tensor
    x1.append(torch.Tensor(x_gt))
    x2.append(torch.Tensor(x_synth))

    return print(batch_mel_cepstral_distortion(x1, x2, config))   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to calculate the Mel Cepstral Distortion between the ground truth and the synthesized audios.')
    parser.add_argument("--gt_path", required = True, type=str, help = 'Path to corresponding ground truth audio in .wav format')
    parser.add_argument("--synth_path", required = True, type=str, help = 'Path to corresponding synthesized audio in .wav format')

    args = parser.parse_args()

    gt_path = args.gt_path
    synth_path = args.synth_path

    main(gt_path, synth_path)