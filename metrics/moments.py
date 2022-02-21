import argparse
import librosa
import numpy as np
from audio.pitch import yin, dio
from config.global_config import GlobalConfig
from scipy.stats import tstd, skew, kurtosis

def estimate_moments(pitch_gt, pitch_skew):
    gt_moments = {'std':tstd(pitch_gt), 'skew':skew(pitch_gt), 'kurt':kurtosis(pitch_gt)}
    synth_moments = {'std':tstd(pitch_skew), 'skew':skew(pitch_skew), 'kurt':kurtosis(pitch_skew)}
    return {'GT': gt_moments, 'SYNTH':synth_moments}

def main(gt_path, synth_path, pitch_algorithm):
    
    # Load Audio
    x_gt, _ = librosa.load(gt_path)
    x_synth, _ = librosa.load(synth_path)

    # Load Config
    config = GlobalConfig()

    # Compute Pitch
    pitch_gt = eval(pitch_algorithm)(x_gt, config)['pitches']
    pitch_synth = eval(pitch_algorithm)(x_synth, config)['pitches']

    # Compute Moments
    return print(estimate_moments(pitch_gt, pitch_synth))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to estimate the Statistical Moments of the ground truth and the synthesized audios.')
    parser.add_argument("--gt_path", required = True, type=str, help = 'Path to corresponding ground truth audio in .wav format')
    parser.add_argument("--synth_path", required = True, type=str, help = 'Path to corresponding synthesized audio in .wav format')
    parser.add_argument("--pitch_algorithm", required = True, type=str, choices = ["dio", "yin"], help = 'Choose method of computing pitch')

    args = parser.parse_args()

    gt_path = args.gt_path
    synth_path = args.synth_path
    pitch_algorithm = args.pitch_algorithm

    main(gt_path, synth_path, pitch_algorithm)