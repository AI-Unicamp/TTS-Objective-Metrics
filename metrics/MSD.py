import argparse
from config.global_config import GlobalConfig
import librosa
import torchaudio
import torch
from metrics.helpers import batch_compute_distortion

# https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/speech_synthesis/utils.py
def batch_mel_spectral_distortion(
    y1, y2, config
):
    """
    https://arxiv.org/pdf/2011.03568.pdf
    Same as Mel Cepstral Distortion, but computed on log-mel spectrograms.
    """
    offset = 1e-6
    return batch_compute_distortion(
        y1, y2, config.sr, lambda y: torch.log(config.mel_fn.to(y1[0].device)(y) + offset).transpose(-1, -2),
        'compute_rms_dist', config.norm_align_type
    )

def main(gt_path, synth_path):
    
    # Load Audio
    x_gt, _ = librosa.load(gt_path)
    x_synth, _ = librosa.load(synth_path)

    # Load Config
    config = GlobalConfig()

    # Initiate Lists of Tensors for Batched DTW
    x1 = []
    x2 = []

    # Append Each Tensor
    x1.append(torch.Tensor(x_gt))
    x2.append(torch.Tensor(x_synth))

    return print(batch_mel_spectral_distortion(x1, x2, config))   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to calculate the Mel Spectral Distortion between the ground truth and the synthesized audios.')
    parser.add_argument("--gt_path", required = True, type=str, help = 'Path to corresponding ground truth audio in .wav format')
    parser.add_argument("--synth_path", required = True, type=str, help = 'Path to corresponding synthesized audio in .wav format')

    args = parser.parse_args()

    gt_path = args.gt_path
    synth_path = args.synth_path

    main(gt_path, synth_path)