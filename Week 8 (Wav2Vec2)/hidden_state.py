"""
Author: Daren
Date: Mar 30, 2024
Description: Python script to extract hidden state output from constructed Wav2Vec2 classifier
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from train import Wav2Vec2Classifer         # Import the constructed Wav2Vec2 classifier
from train import read_data, process_data   # Import functions to tokenize the data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NUM = 2
HIDDEN_LAYER_NODES = 32


# Define a forward hook function that can extract output of intermediate layers
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()
    return hook


def get_hidden_state(data_loader, data_type):
    """
    Evaluate model on either train, test, or validation data
    The best model weights derived from the train_and_val function are loaded.

    :param data_loader: DataLoader containing either train, test, or validation data
    :param data_type:   String of whether the data is train, test, or validation
    """   
    # Load pretrained model weights
    model_filepath = os.path.join(os.getcwd(), "model_weights", f"Wav2Vec2_weights_{MODEL_NUM}.pt")
    model = Wav2Vec2Classifer().to(device)
    model.load_state_dict(torch.load(model_filepath))

    # Set model to evaluation mode
    model.eval()
    
    # Create empty DataFrame to store output values
    initialise_data = {"W2V2" + str(i): [] for i in range(HIDDEN_LAYER_NODES)}
    initialise_data["Label"] = []
    column_headers = list(initialise_data.keys())
    df = pd.DataFrame(initialise_data)

    with tqdm(total=len(data_loader)) as pbar:
        
        # Loop over each batch in the data loader
        for batch in data_loader:
            
            # Extract data for each batch and move them to device
            X_batch_seq, X_batch_mask, Y_batch = batch
            X_batch_seq, X_batch_mask, Y_batch = X_batch_seq.to(device), X_batch_mask.to(device), Y_batch.to(device)

            # Extract the output from 3rd hidden layer
            model.fc3.register_forward_hook(get_activation('fc3'))
            model(X_batch_seq, X_batch_mask)
            hidden_state = activation['fc3']

            # Append hidden state for current batch
            Y_batch = Y_batch.detach().cpu().numpy()[:, np.newaxis]
            append_data = np.hstack((hidden_state, Y_batch))
            temp_df = pd.DataFrame(append_data, columns=column_headers)
            df = pd.concat([df, temp_df], ignore_index=True)

            pbar.update(1)

    # Reset the model back to training mode
    model.train()

    # Define and create directory if does not exist
    output_directory = os.path.join(os.getcwd(), "data", f"hidden{MODEL_NUM}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Store DataFrame as CSV file
    output_filepath = os.path.join(output_directory, f"{data_type}.csv")
    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    print("Step 1 of 4: Load and tokenize data")
    data = read_data()
    train_loader, val_loader, test_loader = process_data(data)

    print("Step 2 of 4: Extract hidden states for training data")
    get_hidden_state(train_loader, "train")

    print("Step 3 of 4: Extract hidden states for validation data")
    get_hidden_state(val_loader, "val")

    print("Step 4 of 4: Extract hidden states for test data")
    get_hidden_state(test_loader, "test")