"""
Author: Daren Tan
Date: Apr 02, 2024
Description: Python script to train and test models on manually extracted features

Code requires the following Python scripts to be in the same directory
- filter.py: Filter out audio files that do not fall within our defined criterias
- preprocessing.py: Resample, filter, and convert audio file to numpy array
- features.py: Manually extract features from audio data
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split

from filter import get_file_paths
from features import extract_features_from_files

def split_dataset():
    """
    Split the overall dataset into train and test sets

    :return: DataFrame containing Features and Labels of the train and test sets
    """
    # Read the features file
    csv_file = os.path.join(os.getcwd(), "features2.csv")
    df = pd.read_csv(csv_file)

    # Split data into train and test set
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4248)

    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train and evaluate the performance on various non-deep-learning models

    :param X_train: DataFrame containing features of train set
    :param y_train: DataFrame containing labels of train set
    :return:        List of trained models
    """
    # Scale the features to values between 0 and 1
    scaler = MinMaxScaler()

    # Initialise models to evaluate the data on
    trained_models = dict()
    models = {
        "MNB": MultinomialNB(),
        "LOG": LogisticRegression(max_iter=500, random_state=4248),
        "SVC": SVC(random_state=4248),
        "KNN": KNeighborsClassifier()
    }

    print("\nValidation scores of various models")
    for model_name, model in models.items():
        # Using all the features
        # X_scaled = scaler.fit_transform(X_train)

        # Using a reduced set of features
        X_reduced = drop_features(X_train, model_name)
        X_scaled = scaler.fit_transform(X_reduced)
        
        # Perform cross validation of the dataset on the models
        cv_scores = cross_val_score(model, X_scaled, y_train, cv=5, scoring="f1_macro")
        print(f"{model_name}: Mean CV F1-Macro Score = {cv_scores.mean():.5f}")
        trained_models[model_name] = model.fit(X_scaled, y_train)

    # Code Output before feature reduction:
    # Validation scores of various models
    # MNB: Mean CV F1-Macro Score = 0.10978
    # LOG: Mean CV F1-Macro Score = 0.27072
    # SVC: Mean CV F1-Macro Score = 0.31241
    # KNN: Mean CV F1-Macro Score = 0.34353

    # Code Output after feature reduction:
    # Validation scores of various models
    # MNB: Mean CV F1-Macro Score = 0.10905
    # LOG: Mean CV F1-Macro Score = 0.26727
    # SVC: Mean CV F1-Macro Score = 0.31380 [improved]
    # KNN: Mean CV F1-Macro Score = 0.34353

    return trained_models


def test(models, X_test, y_test):
    """
    Test the performance of models on the train set

    :param models:  List of trained models
    :param X_train: DataFrame containing features of test set
    :param y_train: DataFrame containing labels of test set
    """
    # Scale the features to values between 0 and 1
    scaler = MinMaxScaler()

    # Iterate through all models to get performance metric
    print("\nTest scores of various models")
    for model_name, model in models.items():
        # Using all the features
        # X_scaled = scaler.fit_transform(X_test)

        # Using a reduced set of features
        X_reduced = drop_features(X_test, model_name)
        X_scaled = scaler.fit_transform(X_reduced)
        
        y_pred = model.predict(X_scaled)
        score = f1_score(y_test, y_pred, average="macro")
        print(f"{model_name}: F1-Macro = {score:.5f}")

    # Code Output before feature reduction
    # Test scores of various models
    # MNB: F1-Macro = 0.10534
    # LOG: F1-Macro = 0.27010
    # SVC: F1-Macro = 0.30735
    # KNN: F1-Macro = 0.31966

    # Code Output after feature reduction
    # Test scores of various models
    # MNB: F1-Macro = 0.10373
    # LOG: F1-Macro = 0.26776
    # SVC: F1-Macro = 0.30679
    # KNN: F1-Macro = 0.31966


def drop_features(X_data, model_name):
    """
    Remove feature columns from a specified model
    Condition to Remove: If >50% of the permutation instances result in increase in performance

    :param X_data:     DataFrame containing the features
    :param model_name: String containing the 3 letter abbrieviation of the model name
    :return:           Copy of input DataFrame with features removed
    """
    # Uncomment this code if using 13 classes
    # features_to_drop = {
    #     "MNB": ["spectral_rolloff"],
    #     "LOG": ["chroma_2", "chroma_5", "chroma_6", "chroma_8", "chroma_9"],
    #     "SVC": ["chroma_2", "chroma_10"],
    #     "KNN": []
    # }

    # Uncomment this code if using 5 classes
    features_to_drop = {
        "MNB": ["mfcc_1", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12"],
        "LOG": ["zcr", "mfcc_4", "mfcc_7", "chroma_1", "chroma_4"],
        "SVC": [],
        "KNN": []
    }

    # Uncomment this code if not planning to drop any features
    # features_to_drop = {
    #     "MNB": [],
    #     "LOG": [],
    #     "SVC": [],
    #     "KNN": []
    # }
    
    # Drop features of corresponding model
    for key in features_to_drop:
        if key == model_name:
            X_reduced = X_data.copy()
            X_reduced.drop(columns=features_to_drop[key], inplace=True)
    
    return X_reduced


def plot_feature_importance(models, X_train, y_train):
    """
    Plot a figure that shows which figures are important

    :param models:  List of trained models
    :param X_train: DataFrame containing features of train set
    :param y_train: DataFrame containing labels of train set
    """
    # Scale the features to values between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    
    # Indicate the quadrant of the model performance plot
    for model_name in models.keys():
        if model_name == "MNB":
            x, y = 0, 0
        elif model_name == "LOG":
            x, y = 0, 1
        elif model_name == "SVC":
            x, y = 1, 0
        elif model_name == "KNN":
            x, y = 1, 1
        
        # Permute one feature at a time while leaving other features untouched
        perm_importance = permutation_importance(models[model_name], X_scaled, y_train,
                                                 scoring="f1_macro", n_repeats=20, random_state=4248)
        feature_importances = perm_importance.importances
        df = pd.DataFrame(feature_importances.T, columns=X_train.columns)
        df.boxplot(ax=axs[x, y], rot=90)
        axs[x, y].set_title(model_name)
    
    plt.tight_layout()
    plt.savefig("feature_importance2.png")


if __name__ == '__main__':
    print("Step 1 of 4: Filter out the relevant filepaths")
    file_paths = get_file_paths()

    print("Step 2 of 4: Extract features from files")
    extract_features_from_files(file_paths)

    print("Step 3 of 4: Evaluating and Testing models")
    X_train, X_test, y_train, y_test = split_dataset()
    trained_models = train(X_train, y_train)
    test(trained_models, X_test, y_test)

    print("Step 4 of 4: Plotting the feature importance")
    plot_feature_importance(trained_models, X_train, y_train)
