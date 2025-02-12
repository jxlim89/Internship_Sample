"""
Author: Daren Tan
Date: February 25, 2024
Description: Python script to test various models on AFEW-VA dataset
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# Load the pretrained processor and model
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)
# model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=7)   # Model 1 (base)
# model = ViTForImageClassification.from_pretrained("./my_model", num_labels=7)         # Model 2
model = ViTForImageClassification.from_pretrained("./my_model_aug", num_labels=7)       # Model 3


def read_data():
    """
    Read the AFEW-VA data and test it against the trained ViT model
    """
    # Get the JSON filepaths containing extracted facial images from AFEW-VA dataset
    clip_directory = os.path.join(os.path.dirname(os.getcwd()), "Extract AFEW-VA", "out")
    json_paths = [os.path.join(clip_directory, file) for file in os.listdir(clip_directory)]
    
    accuracy_list = []
    for json_path in sorted(json_paths):        
        print(f"Processing file: {os.path.basename(json_path)}", end='\r')

        # Read clip data from json save file
        with open(json_path, 'r') as file:
            clip_data = json.load(file)
            clip_id = list(clip_data.keys())[0]
            accuracy = process_data(clip_data[clip_id]['frames'])
            accuracy_list.append(accuracy)
    
    print(np.mean(np.array(accuracy_list)))

    # Save accuracy for each clip into a json file
    with open("accuracy3.json", 'w') as json_file:
        json.dump(accuracy_list, json_file)

def process_data(frames_data):
    """
    Get the predicted label from the frames of each clip
    :param frames_data: Dictionary of frame pixels, and their VA values
    :return:            Accuracy of the predicted label averaged across all frames of the clip
    """
    prediction = []
    actual = []
    for frame in frames_data.keys():        
        # Fit data into 1 of the 4 groups based off valence and arousal
        valence = frames_data[frame]['valence']
        arousal = frames_data[frame]['arousal']
        if valence < 0 and arousal >= 0:
            actual.append(1)
        elif valence >= 0 and arousal >= 0:
            actual.append(2)
        elif valence < 0 and arousal < 0:
            actual.append(3)
        elif valence >= 0 and arousal < 0:
            actual.append(4)
        
        # Get the model's prediction
        pixel_data = np.array(frames_data[frame]['image']).reshape(256, 256, 3)
        image = Image.fromarray(pixel_data.astype('uint8')).convert('L').convert('RGB')
        inputs = processor(image, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        
        # Fit 7 emotions in 4 groups
        if predicted_label in [0, 1, 2]:
            prediction.append(1)
        elif predicted_label in [3, 5]:
            prediction.append(2)
        elif predicted_label == 4:
            prediction.append(3)
        elif predicted_label == 6:
            prediction.append(4)

    # Compute the accuracy of the clip
    correct_count = sum(1 for a, b in zip(prediction, actual) if a == b)
    total_count = len(prediction)
    accuracy = correct_count / total_count
    return accuracy
    

def compute_accuracy():
    """
    Read accuracy scores for JSON file and compute average accuracy
    """
    with open("accuracy1.json", 'r') as json_file:
        accuracy_list = json.load(json_file)
        print(np.mean(np.array(accuracy_list)))
    with open("accuracy2.json", 'r') as json_file:
        accuracy_list = json.load(json_file)
        print(np.mean(np.array(accuracy_list)))
    with open("accuracy3.json", 'r') as json_file:
        accuracy_list = json.load(json_file)
        print(np.mean(np.array(accuracy_list)))

if __name__ == "__main__":
    #read_data()
    compute_accuracy()