import os
import glob
import librosa 

# Load Audios
def read_folder(path):
    out = list()
    for filename in sorted(glob.glob(os.path.join(path, '*.wav'))):
        x, sr = librosa.load(filename)
        out.append(x)
    return out, sr