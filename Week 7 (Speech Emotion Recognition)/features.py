"""
Author: Daren Tan
Date: Apr 02, 2024
Description: Python script to extract audio features

References: https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
"""
import os
import librosa
import numpy as np
import pandas as pd
from preprocessing import full_process

SAMPLE_RATE = 16000
N_FFT = 256

def extract_features_from_files(file_paths):
    """
    Extract features from list of audio files and save them into a CSV file

    :param file_paths: List of file paths of audio files
    """
    # Boolean value to check if DataFrame to store features have been created
    doesDfExists = False

    # Mapping from 13 emotions to 5 categories based on Valence-Arousal chart
    quadrant_mapping = {1:2, 2:4, 3:1, 4:3, 5:3, 6:3, 7:2, 8:1, 9:0, 10:0, 11:3, 12:2}

    # Iterate through all audio files to extract features
    for i, file_path in enumerate(file_paths):
        print(f"Processing file {i} of {len(file_paths)}", end="\r")
        
        # Extract the label from the file name
        label = os.path.basename(file_path).split("-")[2]
        label = quadrant_mapping[int(label)]    # Comment the following line if using default classes
        
        # Run preprocessing on the audio files
        audio_data = full_process(file_path, False)
        
        # Extract the features from the audio data
        features, feature_names = extract_features(audio_data)

        # Create a DataFrame to store features if not already exists
        if not doesDfExists:
            columns = ['Label'] + feature_names
            df = pd.DataFrame(columns=columns)
            doesDfExists = True

        # Add features to DataFrame
        row_data = [str(label)] + features
        df.loc[len(df.index)] = row_data

    # Save features DataFrame into a CSV file
    save_file = os.path.join(os.getcwd(), "features2.csv")
    df.to_csv(save_file, index=False)


def extract_features(audio_data):
    """
    Extract features from audio data

    :param audio_data: NumPy array of the audio
    :return:           List of features and list of their corresponding feature names
    """
    features = []
    features.append(extract_zcr(audio_data))
    features.append(extract_energy(audio_data))
    features.append(extract_entropy(audio_data))
    features.append(extract_spectral_energy(audio_data))
    features.append(extract_spectral_flux(audio_data))
    features.append(extract_spectral_rolloff(audio_data))
    features.extend(extract_mfcc(audio_data))
    features.extend(extract_chroma(audio_data))
    features.append(extract_chroma_deviation(audio_data))

    feature_names = []
    feature_names.append("zcr")
    feature_names.append("energy")
    feature_names.append("entropy")
    feature_names.append("spectral_energy")
    feature_names.append("spectral_flux")
    feature_names.append("spectral_rolloff")
    feature_names.extend([f"mfcc_{i}" for i in range(13)])
    feature_names.extend([f"chroma_{i}" for i in range(12)])
    feature_names.append("chroma_deviation")

    return features, feature_names

def extract_zcr(audio_data):
    return librosa.feature.zero_crossing_rate(audio_data).mean()

def extract_energy(audio_data):
    return librosa.feature.rms(y=audio_data).mean()

def extract_entropy(audio_data):
    energy = librosa.feature.rms(y=audio_data)
    hist, _ = np.histogram(energy.flatten(), bins=50)
    prob = hist / np.sum(hist)
    entropy_val = -np.sum(prob * np.log2(prob + 1e-12))
    return entropy_val

def extract_spectral_energy(audio_data):
    stft = librosa.stft(y=audio_data)
    squared_magnitudes = np.abs(stft)**2
    spectral_energy = np.sum(squared_magnitudes, axis=1)
    return spectral_energy.mean()

def extract_spectral_flux(audio_data):
    stft = librosa.stft(y=audio_data, n_fft=N_FFT)
    squared_magnitudes = np.abs(stft)**2
    spectral_flux = np.sum(np.abs(np.diff(squared_magnitudes, axis=1)), axis=0)
    return spectral_flux.mean()

def extract_spectral_rolloff(audio_data):
    stft = librosa.stft(y=audio_data, n_fft=N_FFT)
    magnitude_spectrum = np.abs(stft)
    cumulative_sum = np.cumsum(magnitude_spectrum, axis=0)
    total_energy = cumulative_sum[-1]
    energy_threshold = 0.85 * total_energy
    rolloff_index = np.argmax(cumulative_sum >= energy_threshold, axis=0)
    spectral_rolloff = librosa.fft_frequencies(sr=SAMPLE_RATE)[rolloff_index]
    return spectral_rolloff.mean()

def extract_mfcc(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def extract_chroma(audio_data):
    chromas = librosa.feature.chroma_stft(y=audio_data, sr=SAMPLE_RATE, n_chroma=12, n_fft=N_FFT)
    return np.mean(chromas, axis=1)

def extract_chroma_deviation(audio_data):
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=SAMPLE_RATE, n_chroma=12, n_fft=N_FFT)
    chroma_deviation = chroma.std(axis=1)
    return chroma_deviation.mean()
