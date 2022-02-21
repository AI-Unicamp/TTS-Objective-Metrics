# TTS Objective Metrics 🎯

This repository comprises a compilation of the objective metrics used in several text-to-speech (TTS) papers.

## Available Metrics
| Metric | Used In |
| ------ | ------ |
| Voicing Decision Error (VDE) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Mellotron](https://arxiv.org/abs/1910.11997)|
| Gross Pitch Error (GPE) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Mellotron](https://arxiv.org/abs/1910.11997)|
| F0 Frame Error (FFE) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Mellotron](https://arxiv.org/abs/1910.11997)|
| Dynamic Time Warping (DTW) | [FastSpeech2](https://arxiv.org/abs/2006.04558) |
| Mel Spectral Distortion (MSD) | [Wave-Tacotron](https://arxiv.org/abs/2011.03568) |
| Mel Cepstral Distortion (MCD) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Wave-Tacotron](https://arxiv.org/abs/2011.03568) |
| Statistical Moments (STD, SKEW, KURT) | [FastSpeech2](https://arxiv.org/abs/2006.04558) |

## Available Pitch Computation
| Alogrithm | Proposed In |
| ------ | ------ |
| YIN | [(Cheveigné and Kawahara, 2002)](http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf) |
| DIO | [(Morise, Kawahara, and Katayose, 2009)](https://www.aes.org/e-lib/browse.cfm?elib=15165)|
| PYIN (Testing) | [(Mauch and Dixon, 2014)](https://ieeexplore.ieee.org/document/6853678) |

## How to Run
First, clone and enter the repo:
```sh
git clone https://github.com/AI-Unicamp/TTS-Objective-Metrics
cd TTS-Objective-Metrics
```

Install dependencies:
```
pip install -r requirements.txt
```

Then, configure the global parameters as you wish in  config-> global.py. If not done, all statistics will be computed with the default parameters already set.

The main usage of the repo is to calculate all available metrics for a batch of (ground truth, synthesized) audio pairs (test/eval set). For this make sure you have the (ground_truth, synthesized) audio pairs names **matching** and in **numbered order, each with the same number of digits of the greatest file** (eg. if there are 100 files, you shall start with 000.wav, if there are [10,99] you shall start with 00.wav). as in:

📂My Audio\
 ┣ 📂Ground Truths\
 ┃ ┣ 📜00.wav\
 ┃ ┣ 📜01.wav\
 ┃ ┣ ...\
 ┣ 📂Synthesizeds\
 ┃ ┣ 📜00.wav\
 ┃ ┣ 📜01.wav\
 ┃ ┣ ...

Then, choose one pitch computing algorithm and run the following command:
```sh
  python -m bin.compute_metrics --gt_folderpath 'My Audio\Ground Truths' --synth_path 'My Audio\Synthesizeds' --pitch_algorithm 'yin'
``` 
The result will be saved in a file named metrics.json in the main repo folder: 'TTS Objective Metrics/metrics.json'.

Alternatively, it is possible to calculate a single metric for a pair of (ground truth, synthesized) audio, by choosing an available metric and pitch computation method (yin or dio) with one of the following commands:
```sh
# For DTW, FFE, GPE, VDE, moments
python -m metrics.DTW --gt_path 'ground_truth_audio.wav' --synth_path 'synthesized_audio.wav' --pitch_algorithm 'yin'
```
```
# For MSD or MCD   
python -m metrics.MSD --gt_path 'path_to_ground_truth_audio.wav' --synth_path 'path_to_synthesized_audio.wav'           
```
The result will be displayed in the terminal.

## Repo Organization
📦TTS Objective Metrics\
 ┣ 📂audio\
 ┃ ┣ 📜helpers.py\
 ┃ ┣ 📜pitch.py\
 ┃ ┣ 📜visuals.py\
 ┣ 📂bin\
 ┃ ┣ 📜compute_metrics.py\
 ┣ 📂config\
 ┃ ┣ 📜global_config.py\
 ┣ 📂metrics\
 ┃ ┣ 📜dists.py\
 ┃ ┣ 📜DTW.py\
 ┃ ┣ 📜FFE.py\
 ┃ ┣ 📜GPE.py\
 ┃ ┣ 📜helpers.py\
 ┃ ┣ 📜MCD.py\
 ┃ ┣ 📜MSD.py\
 ┃ ┣ 📜VDE.py\
 ┣ 📜README.md

## How to Contribute
As the repo is still in its infancy, feel free to either open an issue, discussion or send a pull request, or even contact us by e-mail.

## Authors
- Leonardo B. de M. M. Marques (leonardoboulitreau@gmail.com)
- Lucas Hideki Ueda (lucashueda@gmail.com)

## Github references
- [Coqui-AI](https://github.com/coqui-ai/TTS)
- [Facebook Fairseq](https://github.com/pytorch/fairseq)
- [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)
- [NVIDIA Mellotron](https://github.com/NVIDIA/mellotron/tree/d5362ccae23984f323e3cb024a01ec1de0493aff)
- [MAPS](https://github.com/bastibe/MAPS-Scripts)
- [Yin](https://github.com/patriceguyot/Yin)

All references are listened on top of the used code itself.
