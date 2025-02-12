"""
Author: Daren Tan
Date: March 20, 2024
Description: Python script to extract hidden states from various models

Acknowledgement of models used:
1) VideoMAE - https://huggingface.co/docs/transformers/main/en/model_doc/videomae
2) TimeSformer - https://huggingface.co/docs/transformers/main/en/model_doc/timesformer
3) ViViT - https://huggingface.co/docs/transformers/main/en/model_doc/vivit
"""

import os
import json
import torch
import numpy as np

## Import for VideoMAE (Model 1)
from transformers import AutoImageProcessor, VideoMAEModel
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

## Imports for TimeSformer (Model 2)
# from transformers import AutoImageProcessor, TimesformerModel
# image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
# model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

## Imports for ViViT (Model 3)  
# from transformers import VivitImageProcessor, VivitModel
# image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")

# Indicate which model is being trained
DATA_NUM = "1"

# Number of frames per sequence the model accepts (1: 16; 2: 8; 3: 32)
SEQ_LEN = 16


def create_save_directory():
    """
    Create directories to save the hidden states of the models
    Directory: dataset -> data{DATA_NUM} -> train/test
    """
    
    # Create the base directory for all datasets
    save_dir = os.path.join(os.getcwd(), "dataset")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create the directory to save training data
    train_dir = os.path.join(save_dir, f"data{DATA_NUM}", "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    # Create the directory to save testing data
    test_dir = os.path.join(save_dir, f"data{DATA_NUM}", "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)


def read_file():
    """
    Read in processed video frames, run it through a model, and get hidden state outputs
    """
    
    # Get file paths for all videos
    video_dir = os.path.join(os.path.dirname(os.getcwd()), "Extract AFEW-VA", "out")
    file_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]   
    
    # Initialise train and test counters to determine which set to assign output data
    train_count = 0
    test_count = 0

    # Iterate through all videos to extract hidden states
    for file_path in sorted(file_paths):
        # Extract video id from filename
        file_id = os.path.basename(file_path)[:3]
        print(f"Currently processing video {file_id} of 600", end="\r")
        
        # Load the JSON file containing the video frames
        with open(file_path, "r") as file:
            data = json.load(file)
            frame_ids = data[file_id]["frames"].keys()

            # Initialise list to store video + valence-arousal data
            video = []
            va = []

            # Iterate through all frames for a video
            for frame_id in frame_ids:
                # Save frame into video list
                image = np.array(data[file_id]["frames"][frame_id]["image"]).reshape(256, 256, 3)
                video.append(image)
                
                # Save valence-arousal data into VA list
                valence = round(data[file_id]["frames"][frame_id]["valence"], 1)
                arousal = round(data[file_id]["frames"][frame_id]["arousal"], 1)
                va.append((valence, arousal))

            # Extract hidden states from video
            features = get_features(video)

            # Split features into train-test set into roughly 8-2 ratio
            if train_count <= test_count * 4:
                curr_set = "train"
                train_count += len(frame_ids)
            else:
                curr_set = "test"
                test_count += len(frame_ids)

            # Save hidden features into their relevant dataset (train or test)
            save_dir = os.path.join(os.getcwd(), "dataset", f"data{DATA_NUM}", curr_set)
            for i, sequence in enumerate(features):
                file_name = f"{file_id}_{str(i).zfill(3)}_{va[i][0]}_{va[i][1]}"
                save_path = os.path.join(save_dir, file_name)
                np.save(save_path, sequence)

            
def get_features(video):
    """
    Construct sequences from video frames, and extract hidden states from sequence

    :param video: List containing numpy arrays of video frames
    :return:      List of hidden state of model after being fed the sequences
    """

    # Initialise a list to save the hidden states
    features = []

    # Iterate through each frame of the video
    num_frames = len(video)
    for frame in range(num_frames):
        
        # Calculate the sequence index
        values_lower = [max(frame - i, 0) for i in range(int(SEQ_LEN / 2) - 1, 0, -1)]
        value_x = frame
        values_upper = [min(frame + i, num_frames - 1) for i in range(1, int(SEQ_LEN / 2) + 1)]
        seq_idx = values_lower + [value_x] + values_upper

        # Build sequence
        sequence = [video[i] for i in seq_idx]
        sequence = np.transpose(np.array(sequence), (0, 3, 1, 2))
        
        # Extract hidden state + reduce size
        inputs = image_processor(list(sequence), return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        hidden_features = torch.mean(last_hidden_state, dim=1)
        features.append(hidden_features.detach().numpy())

    return features


if __name__ == "__main__":
    create_save_directory()
    read_file()