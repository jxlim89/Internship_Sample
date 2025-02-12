import os
import cv2
import glob
import json
import numpy as np
from sklearn.model_selection import train_test_split

category = {1: "train", 2: "val", 3: "test"}
sequence_size = 7   # Must be an odd number

def read_files():
    """
    Read the video files (JSON) and create the corresponding video sequences
    """
    # Get the video data stored in JSON files
    images_directory = os.path.join(os.path.dirname(os.getcwd()), "Extract AFEW-VA", "out")
    images_filepath = os.path.join(images_directory, "*.json")
    
    # Determine how to split video into test-train-val
    num_clips = len(glob.glob(images_filepath))
    split_list = get_split_list(num_clips)

    # For each video clip, create sequence clips of length sequence_size
    clip_va = dict()
    for filepath in sorted(glob.glob(images_filepath)):
        with open(filepath, "r") as json_file:
            clip_data = json.load(json_file)
            clip_id = list(clip_data.keys())[0]
            va_values = make_video(clip_data[clip_id]['frames'], clip_id, split_list[int(clip_id)])
            clip_va[clip_id] = va_values

    # Save the valence-arousal values of the clips
    va_savepath = os.path.join(os.getcwd(), "AFEW-VA Videos", f"va_out.json")
    with open(va_savepath, "w") as json_file:
        json_file.dump(clip_va, json_file)


def make_video(frames_data, clip_id, set_id):
    """
    Generate sequence clips for all frames of the video clip
    :param frames_data: Dictionary containing all frames of a video clip
    :clip_id:           Video clip number
    :set_id:            ID of the set (train-val-test) the video clip belongs to
    :return:            List of valence-arousal values for each frame of the video clip
    """
    va_array = []

    # Construct a sequence where the current frame is the centre frame of the sequence
    for frame in frames_data.keys():
        print(f"Processing Video {clip_id} Frame {frame}", end="\r")
        valence = frames_data[frame]['valence']
        arousal = frames_data[frame]['arousal']
        quadrant = get_va_quadrant(valence, arousal)
        
        output_file = os.path.join(os.getcwd(), "AFEW-VA Videos", category[set_id], str(quadrant), f"{clip_id}_{frame}.avi")
        make_sequence(frames_data, frame, output_file)
        
        va_array.append((valence, arousal))
        
    return va_array


def get_va_quadrant(valence, arousal):
    """
    Determine which quadrant a frame belongs to
    :param valence: valence value
    :param arousal: arousal value
    :return:        quadrant corresponding to valence-arousal value
    """
    if valence < 0 and arousal >= 0:
        return 1
    elif valence >= 0 and arousal >= 0:
        return 2
    elif valence < 0 and arousal < 0:
        return 3
    elif valence >= 0 and arousal < 0:
        return 4


def make_sequence(frames_data, frame, output_file):
    """
    Generate sequence clips for 1 frame of the video clip
    :param frames_data: Dictionary containing all frames of a video clip
    :param frame:       Frame id of the video clip
    :param output_file: File path to save the sequence clip  
    """
    # Create a sequence for frame_id 
    num_frames = len(frames_data.keys()) - 1
    half_sequence = int((sequence_size - 1) / 2)
    sequence_idx = [max(0, int(frame) - i) for i in range(half_sequence, 0, -1)] + [int(frame)] + \
                   [min(num_frames, int(frame) + i) for i in range(1, half_sequence + 1)]
    sequence_str_idx = [str(num).zfill(5) for num in sequence_idx]
    
    # Write frames of that sequence into an AVI file
    videoWriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), 25, (256, 256))
    for idx in sequence_str_idx:
        pixel_data = np.array(frames_data[idx]['image']).reshape(256, 256, 3).astype("uint8")
        videoWriter.write(pixel_data)
    videoWriter.release()


def get_split_list(num_clips):
    """
    Create a list to specify how to split the clips into the train-val-test sets
    :param num_clips: Total number of video clips
    :return:          List where the value at each index indicates the set a clip belongs to
    """
    # Define the size of training, validation, test sets
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    train_size = int(train_ratio * num_clips)
    val_size = int(val_ratio * num_clips)
    test_size = num_clips - train_size - val_size

    # Perform the split based off the clip_ids
    clip_id_list = [i for i in range(num_clips)]
    train_set, remaining_set = train_test_split(clip_id_list, train_size=train_size, random_state=42)
    val_set, test_set = train_test_split(remaining_set, test_size=test_size, random_state=42)

    # Create a split list that determines which set a clip belongs to
    split_list = []
    for clip_id in clip_id_list:
        if clip_id in train_set:
            split_list.append(1)
        elif clip_id in val_set:
            split_list.append(2)
        elif clip_id in test_set:
            split_list.append(3)
    
    return split_list


if __name__ == "__main__":
    read_files()