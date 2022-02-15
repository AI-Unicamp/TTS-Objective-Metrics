import argparse
from distutils.command.config import config
import librosa
import numpy as np
from audio.pitch import yin, dio
from config.global_config import GlobalConfig
from metrics.helpers import same_t_in_true_and_est, voicing_decision_error_frames

# https://github.com/bastibe/MAPS-Scripts/blob/master/helper.py
@same_t_in_true_and_est
def voicing_decision_error(true_t, true_f, est_t, est_f):
    """Percentages of frames that contain error in the voicing
    decision, according to the reference.
    """
    vde_frames = voicing_decision_error_frames(
        true_t, true_f, est_t, est_f
    )
    return np.sum(vde_frames) / (len(true_t))

def main(gt_path, synth_path, pitch_algorithm):
    
    # Load Audio
    x_gt, _ = librosa.load(gt_path)
    x_synth, _ = librosa.load(synth_path)

    # Load Config
    config = GlobalConfig()
    
    # Compute Pitch
    pitch_gt = eval(pitch_algorithm)(x_gt, config)
    pitch_synth = eval(pitch_algorithm)(x_synth, config)

    return print(voicing_decision_error(np.array(pitch_gt['times']), np.array(pitch_gt['pitches']), np.array(pitch_synth['times']), np.array(pitch_synth['pitches'])))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to calculate the Voicing Decision Error between the ground truth and the synthesized audios.')
    parser.add_argument("--gt_path", required = True, type=str, help = 'Path to corresponding ground truth audio in .wav format')
    parser.add_argument("--synth_path", required = True, type=str, help = 'Path to corresponding synthesized audio in .wav format')
    parser.add_argument("--pitch_algorithm", required = True, type=str, choices = ["dio", "yin", "pyin"], help = 'Choose method of computing pitch')

    args = parser.parse_args()

    gt_path = args.gt_path
    synth_path = args.synth_path
    pitch_algorithm = args.pitch_algorithm

    main(gt_path, synth_path, pitch_algorithm)