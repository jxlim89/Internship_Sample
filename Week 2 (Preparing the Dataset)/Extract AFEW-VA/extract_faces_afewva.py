"""
The purpose of this code is to extract faces from each frame of the AFEW-VA dataset.
Each extracted face will be tagged with the corresponding valence and arousal levels.

Instructions:
1) Note that there are 600 clips in total, saved into 12 zip files with 50 clips each.
   Unzip all files and save the folders in the same directory
2) Place this python file into that same directory
3) JSON file containing face pixel data + VA levels will be generated in same directory

Proposed Algorithm:
1) Iterate though current directory for all clips
2) For each clip, iterate through all the frames
3) For each frame, run the face detection algorithm to extract cropped facial images
4) If more than 1 face detected, use facial landmarks to determine correct face
5) Replace facial landmark in dictionary with pixel data of facial image
6) For each dictionary, assigned it to a clip_id as a new key (creates a new dict)
7) Export the new dictionary into a json file

Due to the time it takes to process each clip, we set "checkpoints" so that for every
clip we have, 1 JSON file will be created.
"""

import json
import os
import cv2
import face_detection
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Initialise the face detector using Dual Shot Face Detector (DSFD)
face_detector = face_detection.build_detector("DSFDDetector", max_resolution=1080)

def face_crop(img):
    """
    Crop face(s) from image
    :param img: Image to be processed for facial detection
    :return: List containing cropped face(s)
    """
    # Run the face detector on the image
    faces = face_detector.detect(img)

    crop_faces = []
    for face in faces:

        # Accept as face if algorithm is 90% confident (can be changed)
        confidence = face[4]
        if confidence > 0.90:
            
            # Extract the coordinates for the rectangular border
            border_pts = [abs(int(_)) for _ in face[:4]]
            xmin, ymin, xmax, ymax = border_pts
            
            # Crop image and resized to 120 x 120 pixels (maintaining aspect ratio)
            canvas = Image.new("RGB", (120, 120), (0, 0, 0))
            image = Image.fromarray(img[ymin:ymax, xmin:xmax], 'RGB')
            image.thumbnail((120, 120), Image.LANCZOS)
            
            # Pad the empty space with black pixels
            paste_x = (120 - (image.width)) // 2
            paste_y = (120 - (image.height)) // 2
            canvas.paste(image, (paste_x, paste_y))
            resized_face = list(canvas.getdata())

            crop_faces.append((resized_face, border_pts))
    
    return crop_faces


def process_multiple_faces(faces, landmarks):
    """
    Given multiple faces and facial landmarks belonging to a single face,
    find out which face does the landmarks belong to
    :param faces:     List of face pixel data
    :param landmarks: List containing x-y coordinates of facial landmarks
    :return: Facial pixel data corresponding to most likely face
    """
    # Counter to attribute landmark to a face
    landmark_cts = [0 for _ in range(len(faces))]
    
    for landmark in landmarks:
        for i, face in enumerate(faces):
            
            # Check if landmark falls within facial image coordinate boundary
            xmin, ymin, xmax, ymax = face[1]
            if xmin <= landmark[0] <= xmax and ymin <= landmark[1] <= ymax:
                landmark_cts[i] += 1
    
    # Pick out the face with the highest landmark counters
    likely_face = faces[landmark_cts.index(max(landmark_cts))]

    return likely_face


def image_processing(dir):
    """
    Given a directory of frames for 1 clip, extract the facial data + VA levels
    :param dir: Directory containing the frames of 1 clip
    :return: Dictionary of facial data + VA levels
    """
    # Read json file containing information on VA levels + facial landmarks
    json_file_path = os.path.join(dir, f"{os.path.basename(dir)}.json")
    with open(json_file_path, 'r') as read_file:
        read_data = json.load(read_file)

    frame_ids = list(read_data['frames'].keys())
    for frame_id in frame_ids:        
        
        # Read image data and crop the face(s) out
        frame_path =  os.path.join(dir, f"{frame_id}.png")
        frame = cv2.imread(frame_path)
        crop_faces = face_crop(frame)
        
        # If more than 1 face detected, use facial landmarks to determine face of interest
        if len(crop_faces) > 1:
            landmarks = read_data['frames'][frame_id]['landmarks']
            crop_face = process_multiple_faces(crop_faces, landmarks)
        else:
            crop_face = crop_faces[0]
        
        # Add image data and delete landmark data (reduce storage size)
        read_data['frames'][frame_id]['image'] = crop_face[0]
        del read_data['frames'][frame_id]['landmarks']

    return read_data


def extract_info(dir):
    """
    Extract the facial data + VA levels for all clips into a JSON file
    :param dir: Directory containing the folders to be processed
    """
    # Get the filepaths of each clip
    clip_paths = []
    for root, dirs, files in os.walk(dir):
        for d in dirs:
            clip_paths.append(os.path.join(root, d))
    num_dirs = len([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])
    clip_paths = clip_paths[num_dirs:]

    # Create a folder to store the output
    out_dir = os.path.join(os.getcwd(), "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    extracted_data = dict()
    for clip_path in sorted(clip_paths):
        
        
        clip_id = os.path.basename(clip_path)
        
        # Used to skip clips that have already been processed
        if int(clip_id) in list(range(1, 100)):
            continue
        
        # Processing each clip
        print(f"Processing clip {int(clip_id)} of {len(clip_paths)}", end='\r')
        extracted_data[clip_id] = image_processing(clip_path)

        # Save the processed data of each clip into a JSON file
        out_file_name = os.path.join("out", f"{int(clip_id)}_out.json")
        with open(out_file_name, 'w') as json_file:
            json.dump(extracted_data, json_file)
        extracted_data = dict()


if __name__ == "__main__":
    extract_info(os.getcwd())