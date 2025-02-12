"""
Author: Daren Tan
Date: March 20, 2024
Description: Python script to combine numpy arrays from generate_sequence.py into CSV file
"""

import os
import numpy as np
import pandas as pd

DATASET = "test"

# Get the directory containing files to combine
storage_path = os.getcwd()
directory = os.path.join(storage_path, "dataset", "data3", DATASET)

# Initialize an empty list to store arrays
data_arrays = []
vas = []

# Iterate through files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.npy'):
        # Load each .npy file
        filepath = os.path.join(directory, filename)
        array_data = np.load(filepath)
        
        # Append the array to the list
        data_arrays.append(array_data)

        # Append valence and arousal to the list
        basename = os.path.basename(filepath)
        file_details = basename.split("_")
        valence = float(file_details[2])
        arousal = float(file_details[3][:-4])
        vas.append((valence, arousal))

# Concatenate a list of numpy arrays into a Pandas DataFrame and save as CSV
combined_array = np.concatenate(data_arrays, axis=0)
df = pd.DataFrame(combined_array)
test_filepath = os.path.join(storage_path, f"{DATASET}.csv")
df.to_csv(test_filepath)
print(df)

# Save valence-arousal labels as CSV
label_df = pd.DataFrame(vas, columns=['Valence', 'Arousal'])
label_filepath = os.path.join(storage_path, f"{DATASET}_label.csv")
label_df.to_csv(label_filepath)
print(label_df)
