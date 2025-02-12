"""
Author: Daren Tan
Date: March 20, 2024
Description: Python script to train and test MLP model on generated hidden states
"""

import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset

# Indicate which model was used to generate the hidden states
DATA_NUM = "1"

# Indicate the number to save the current MLP model as
MODEL_NUM = "1_0"


class MLP(nn.Module):
    def __init__(self, input_size=1568, hidden_size=[256, 128, 64, 32], dropout_prob=[0.8, 0.7, 0.6, 0.5]):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[0]),

            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[1]),

            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[2]),

            nn.Linear(hidden_size[2], hidden_size[3]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[3]),

            nn.Linear(hidden_size[3], 2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class PyTorchEstimator(nn.Module):
    def __init__(self, model):
        super(PyTorchEstimator, self).__init__()
        self.model = model

    def fit(self, X, y):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(5):  # Train for a fixed number of epochs
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Print average loss for each epoch
            print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')

    def predict(self, X):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=32)
        predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                outputs = self.model(inputs[0])
                predictions.extend(outputs.numpy())
        return np.array(predictions)

    def get_params(self, deep=True):
        return {'model': self.model}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Calculate the Concordance Correlation Coefficient (CCC) score between 2 inputs
    - Instead of returning the valence and arousal CCC score separately, the mean is calculated
    - This works because the goal is to maximise both the valence and arousal CCC score

    :param y_true: Ground truth valence and arousal values
    :param y_pred: Predicted valence and arousal values
    :return:       Mean of valence and arousal score
    """
    
    # Initiate list to save computed CCC scores for Valence and Arousal
    ccc_scores = []
    
    # Calculate valence CCC score followed by arousal CCC score
    for i, dimension in enumerate(["Valence", "Arousal"]):
        data_true = y_true[dimension].values
        data_pred = y_pred[:, i]
        
        cor = np.corrcoef(data_true, data_pred)[0][1]
        mean_true, mean_pred = np.mean(data_true), np.mean(data_pred)
        var_true, var_pred = np.var(data_true), np.var(data_pred)
        sd_true, sd_pred = np.std(data_true), np.std(data_pred)
        
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        ccc = numerator / denominator
        ccc_scores.append(ccc)
    
    # Get the mean CCC score for all values
    return np.mean(np.array(ccc_scores))

####################################################################################################
# Testing the Model
####################################################################################################

def train_model():
    """
    Train the MLP model with hidden state inputs
    """

    # Define the model
    model = MLP(input_size=100)
    estimator = PyTorchEstimator(model=model)
    
    # Define the parameters to test on
    param_grid = {
        "hidden_size": [[1024, 512, 256, 128], [512, 256, 128, 64], [256, 128, 64, 32]],
        "dropout_prob": [[0.8, 0.7, 0.6, 0.5], [0.8, 0.75, 0.7, 0.65], [0.65, 0.6, 0.55, 0.5], [0.7, 0.65, 0.6, 0.55]]
    }

    # Perform Grid Search over the given parameters
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring=make_scorer(concordance_correlation_coefficient), verbose=10)

    # Extract the data from files
    X_train = pd.read_csv(os.path.join(os.getcwd(), "dataset", f"data{DATA_NUM}", "train.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(os.getcwd(), "dataset", f"data{DATA_NUM}", "train_label.csv"), index_col=0)

    # Apply PCA on X_train (Comment out to not use PCA - remove input size to MLP)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_train)
    pca = PCA(n_components=100, random_state=4248)
    pca_result = pca.fit_transform(scaled_data)
    X_train = pd.DataFrame(data=pca_result)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Save the best model
    model_dir = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model = grid_search.best_estimator_
    model_path = os.path.join(model_dir, f"model{MODEL_NUM}.pkl")
    joblib.dump(best_model, model_path)

    # Save the best parameters
    best_params = grid_search.best_params_
    params_path = os.path.join(model_dir, f"best_params{MODEL_NUM}.txt")
    with open(params_path, "w") as file:
        file.write(json.dumps(best_params))


def test_model():
    """
    Test the trained MLP model
    """
    
    # Load the model from the training step
    model_dir = os.path.join(os.getcwd(), "model")
    model_path = os.path.join(model_dir, f"model{MODEL_NUM}.pkl")
    best_model = joblib.load(model_path)
    
    # Extract the data from files
    X_test = pd.read_csv(os.path.join(os.getcwd(), "dataset", f"data{DATA_NUM}", "test.csv"), index_col=0)
    y_test = pd.read_csv(os.path.join(os.getcwd(), "dataset", f"data{DATA_NUM}", "test_label.csv"), index_col=0)

    # Apply PCA on X_test (Comment out to not use PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_test)
    pca = PCA(n_components=100, random_state=4248)
    pca_result = pca.fit_transform(scaled_data)
    X_test = pd.DataFrame(data=pca_result)
    
    # Get ccc score
    y_predict = best_model.predict(X_test)
    ccc = concordance_correlation_coefficient(y_test, y_predict)
    print(ccc)

if __name__ == '__main__':
    train_model()
    test_model()