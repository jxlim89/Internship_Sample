"""
Author: Daren Tan
Date: Apr 02, 2024
Description: Python script to run preprocessing on audio files

References: https://www.geeksforgeeks.org/preprocessing-the-audio-dataset/
"""
import librosa
import numpy as np
from scipy.signal import butter, filtfilt

def full_process(audio_path, isModel=True):
    resampled_audio, sample_rate = resample_audio(audio_path)
    filtered_audio = butter_lowpass_filter(resampled_audio, sample_rate)
    if isModel:
        return convert_to_model_input(filtered_audio)
    return filtered_audio

# Resample the audio
def resample_audio(audio_path, target_sr=16000):
    y, sr = librosa.load(audio_path, sr=target_sr)
    return y, sr

# Apply filtering
def butter_lowpass_filter(data, sample_rate, cutoff_freq=4000, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Convert audio data to the model’s expected input
def convert_to_model_input(y, target_length=16000):
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y