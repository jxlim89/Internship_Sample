"""
Author: Daren Tan
Date: Apr 02, 2024
Description: Python script to filter files
"""
import os
import wave

DATA_DIRECTORY = "/mnt/store/tjunheng/ASVP-ESD-Update/Audio"

def get_file_paths(data_dir=DATA_DIRECTORY):
    """
    Get the file paths of remaining audio files after filtering process

    :param data_dir: Directory where the audio files are stored
    :return:         List of full file paths of remaining files in alphabetical order
    """
    # Filter out features based on various identifiers
    filter_identifier_condition = [
        lambda x: x[2] != "13",                     # Remove "breath" as an emotion
        lambda x: x[3] in ["01", "02"],             # Remove unknown intensity value
        lambda x: x[6] in ["01", "02", "03", "04"], # Remove actors of unknown age
        lambda x: x[7] in ["01", "02", "03"]        # Remove unknown source of downloads
    ]

    # Filter out features based on audio clip duration
    filter_wav_condition = [
        lambda x: x >= 1,   # Ensure clip is at least 1 second long
        lambda x: x <= 10   # Ensure clip is no more than 10 seconds long
    ]

    # Iterate through all the clips
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