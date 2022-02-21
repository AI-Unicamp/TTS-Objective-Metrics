# Basic Imports
import argparse
import json
import numpy as np
import torch

# Import Audio Loader
from audio.helpers import read_folder

# Import Pitch Computation
from audio.pitch import dio, yin

# Import Config
from config.global_config import GlobalConfig

# Import Metrics
from metrics.VDE import voicing_decision_error
from metrics.GPE import gross_pitch_error
from metrics.FFE import f0_frame_error
from metrics.DTW import batch_dynamic_time_warping
from metrics.MSD import batch_mel_spectral_distortion
from metrics.MCD import batch_mel_cepstral_distortion
from metrics.moments import estimate_moments

# Import Basic Stats
from metrics.helpers import add_basic_stats

# Get Input Args
parser = argparse.ArgumentParser(description='Script to calculate the all repo metrics between the ground truth and the synthesized audios.')
parser.add_argument("--gt_folder_path", required = True, type=str, help = 'Path to the folder containing all ground truth audio in .wav format')
parser.add_argument("--synth_folder_path", required = True, type=str, help = 'Path to the folder containing all respective synthesized audio in .wav format')
parser.add_argument("--pitch_algorithm", required = True, type=str, choices = ["dio", "yin"], help = 'Choose method of computing pitch')
args = parser.parse_args()
gt_folder_path = args.gt_folder_path
synth_folder_path = args.synth_folder_path
pitch_algorithm = args.pitch_algorithm

# Import Audios
refs, _ = read_folder(gt_folder_path)
synths, _ = read_folder(synth_folder_path)
refs_tensor = [torch.Tensor(item) for item in refs]
synths_tensor = [torch.Tensor(item) for item in synths]

# Define Configurations
config = GlobalConfig()

# Compute Pitch
refs_pitch = [eval(pitch_algorithm)(item, config) for item in refs]
synths_pitch = [eval(pitch_algorithm)(item, config) for item in synths]
refs_pitch_tensor = [torch.Tensor(item['pitches']).unsqueeze_(1) for item in refs_pitch]
synths_pitch_tensor = [torch.Tensor(item['pitches']).unsqueeze_(1) for item in synths_pitch]

# Pitch DistributionS
refs_dist = [item for sublist in refs_pitch for item in sublist['pitches']]
synths_dist = [item for sublist in synths_pitch for item in sublist['pitches']]

# Pairwise Metrics
VDE = {}
GPE = {}
FFE = {}
for i in range(len(refs)):
    VDE[i] = voicing_decision_error(refs_pitch[i]['times'], refs_pitch[i]['pitches'], synths_pitch[i]['times'], synths_pitch[i]['pitches'])
    GPE[i] = gross_pitch_error(refs_pitch[i]['times'], refs_pitch[i]['pitches'], synths_pitch[i]['times'], synths_pitch[i]['pitches'])
    FFE[i] = f0_frame_error(refs_pitch[i]['times'], refs_pitch[i]['pitches'], synths_pitch[i]['times'], synths_pitch[i]['pitches'])

# Batched Metrics
DTW = {v:k for v,k in enumerate(batch_dynamic_time_warping(refs_pitch_tensor, synths_pitch_tensor, config.dist_fn, config.norm_align_type)['norm_align_costs'])}  
MSD = {v:k for v,k in enumerate(batch_mel_spectral_distortion(refs_tensor, synths_tensor, config))}
MCD = {v:k for v,k in enumerate(batch_mel_cepstral_distortion(refs_tensor, synths_tensor, config))}

# Distribution Metrics
MOMENTS = estimate_moments(refs_dist, synths_dist)

# Compute Basic Stats and Store
stats = {}
stats['VDE'] = add_basic_stats(VDE)
stats['GPE'] = add_basic_stats(GPE)
stats['FFE'] = add_basic_stats(FFE)
stats['DTW'] = add_basic_stats(DTW)
stats['MSD'] = add_basic_stats(MSD)
stats['MCD'] = add_basic_stats(MCD)
stats['Moments'] = MOMENTS

# Create Output File
tts_objective_metrics = {'tts_objecive_metrics': stats}
json_object = json.dumps(tts_objective_metrics, indent = 4)
with open("metrics.json", "w") as outfile:
    outfile.write(json_object)
