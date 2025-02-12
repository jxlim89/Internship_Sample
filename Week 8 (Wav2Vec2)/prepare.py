"""
Author: Daren Tan
Date: Apr 12, 2024
Description: Python script to train and test models on manually extracted features
"""
import os
import wave
import librosa
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

DATASET = 2
DATA_DIRECTORY = "/mnt/store/tjunheng/ASVP-ESD-Update/Audio"
SAVE_DIRECTORY = "/mnt/store/tjunheng/ASVP-ESD-Update/Extracted"

def construct_path_df():   
    """
    Create a DataFrame that contains the path of audio files

    :return: DataFrame that contains the filename, filepath, and labelled class
    """
    # Define the directory to save the DataFrame containing the file paths
    df_path = os.path.join(SAVE_DIRECTORY + "/paths.csv")
    
    # If the save file already exists, read and return the DataFrame instead
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        return df

    data = []
    actor_counts = dict()
    
    # Uncomment if mapping from 12 emotions to 5 classes based on Valence-Arousal chart
    quadrant_mapping = {1:2, 2:4, 3:1, 4:3, 5:3, 6:3, 7:2, 8:1, 9:0, 10:0, 11:3, 12:2}
    
    # Obtain a list of file paths
    file_paths = get_file_paths()
    
    for file_path in file_paths:       

        # Extract class label from file name
        label = int(os.path.basename(file_path).split("-")[2])
        label = quadrant_mapping[label]     # Uncomment this line if using 5 class labels

        # Extract actor id from file name
        actor = int(os.path.basename(file_path).split("-")[5])

        # Create count label for each audio file per actor
        if actor in actor_counts:
            actor_counts[actor] += 1
            count = actor_counts[actor]
        else:
            actor_counts[actor] = 0
            count = 0

        # Build file name from actor id, actor clip count, and its label
        name = f"{str(actor)}_{str(count)} ({str(label)})"

        # Add extracted information into a list
        data.append([name, file_path, label])
    
    # Convert list of data into DataFrame and save as a CSV file
    df = pd.DataFrame(data, columns=["name", "path", "emotion"])
    df.to_csv(df_path, index=False)

    return df


def get_file_paths(data_dir=DATA_DIRECTORY):
    """
    Get the file paths of remaining audio files after filtering process

    :param data_dir: Directory where the audio files are stored
    :return:         List of full file paths of remaining files in alphabetical order
    """
    filter_identifier_condition = [
        lambda x: x[2] != "13",                     # Remove "breath" as an emotion
        lambda x: x[3] in ["01", "02"],             # Remove unknown intensity value
        lambda x: x[6] in ["01", "02", "03", "04"], # Remove actors of unknown age
        lambda x: x[7] in ["01", "02", "03"]        # Remove unknown source of downloads
    ]

    filter_wav_condition = [
        lambda x: x >= 1,   # Ensure clip is at least 1 second long
        lambda x: x <= 10   # Ensure clip is no more than 10 seconds long
    ]

    file_paths = []
    for i in range(129):
        print(f"Processing clips for Actor {i}", end="\r")

        audio_directory = os.path.join(data_dir, f"actor_{i}")
        for filename in os.listdir(audio_directory):
            # Extract identifiers from file name
            identifiers = filename[:-4].split("-")
            
            # Filter files which do not satisfy the conditios specified earlier
            if all(condition(identifiers) for condition in filter_identifier_condition):
                wav_duration = get_wav_duration(os.path.join(audio_directory, filename))
                if all(condition(wav_duration) for condition in filter_wav_condition):
                    file_paths.append(os.path.join(audio_directory, filename))
    
    return sorted(file_paths)


def get_wav_duration(file_path):
    """
    Extract the duration of the audio clip

    :param file_path: File path of the audio clip
    :return:          Length of the audio clip in seconds
    """
    with wave.open(file_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration_seconds = num_frames / float(frame_rate)
        
    return duration_seconds


def split_data(df):
    """
    Split a DataFrame into train-test-val sets in a 72-20-8 ratio
    Saves each of the data sets into a CSV file

    :param df: DataFrame that contains the filename, filepath, and labelled class
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=4248, stratify=df["emotion"])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=4248, stratify=train_df["emotion"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(f"{SAVE_DIRECTORY}/train{DATASET}.csv", index=False)
    test_df.to_csv(f"{SAVE_DIRECTORY}/test{DATASET}.csv", index=False)
    val_df.to_csv(f"{SAVE_DIRECTORY}/val{DATASET}.csv", index=False)


def full_process(audio_path):
    resampled_audio, sample_rate = resample_audio(audio_path)
    filtered_audio = butter_lowpass_filter(resampled_audio, sample_rate)
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


if __name__ == '__main__':
    df = construct_path_df()
    split_data(df)
