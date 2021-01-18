# In this script, I'll be extracting features from the music files into arrays to apply ML/DL models

# Importing libraries
import os
import csv
import librosa
import numpy as np

# Extracting features 
# The following features will be extracted:
# 1) Chroma Frequencies
# 2) Spectral Centroid
# 3) Spectral Bandwidth
# 4) Spectral Roll off
# 5) Zero Crossing Rate
# 6) MFCCs (20)
# 7) RMSE
header = 'Filename Chroma_Frequencies Spectral_Centroid Spectral_Bandwidth Spectral_Rolloff Zero_Crossing_Rate RMSE'
for i in range(1, 21):
    header += f' MFCC_{i}'
header += " Label"
header = header.split()

file = open('music_speech_features.csv', 'w', newline = '')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
    
labels = 'music_au speech_au'.split()
for i in labels:
    for name in os.listdir(f'D:/Big Datasets/Music Speech Classification/{i}'):
        filename = f'D:/Big Datasets/Music Speech Classification/{i}/{name}'
        y, sr = librosa.load(filename, mono = True)
        chroma_frequencies = librosa.feature.chroma_stft(y = y, sr = sr)
        spectral_centroid = librosa.feature.spectral_centroid(y = y, sr = sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y = y, sr = sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y = y)
        mfccs = librosa.feature.mfcc(y = y, sr = sr)
        rmse = librosa.feature.rms(y = y)
        appendations = f'{name} {np.mean(chroma_frequencies)} {np.mean(spectral_centroid)} {np.mean(spectral_bandwidth)} {np.mean(spectral_rolloff)} {np.mean(zero_crossing_rate)} {np.mean(rmse)}'
        for j in mfccs:
            appendations += f' {np.mean(j)}'
        appendations += f' {i}'
        
        file = open('music_speech_features.csv', 'a', newline = '')
        with file:
            write = csv.writer(file)
            write.writerow(appendations.split())
        
