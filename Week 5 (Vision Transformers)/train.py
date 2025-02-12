"""
Author: Daren Tan
Date: February 25, 2024
Description: Python script to finetune the ViT Image Classifer on FER2013 dataset

Code adapted from:
- Author  = Nate Raw
- Title   = Fine-Tune ViT for Image Classification with 🤗 Transformers
- Journal = HuggingFace blog
- Date    = February 11, 2022
- url     = https://huggingface.co/blog/fine-tune-vit 
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from evaluate import load
from imgaug import augmenters as iaa
from datasets import Dataset, DatasetDict
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Load the pretrained processor and model
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=7)

def read_data():
    """
    Read facial images from a csv file
    :return: Dataset (in HuggingFace format) of the train, test, and validation data
    """
    # Initialise dictionary for testing and training data (data is saved with train/test label in csv)
    test_dict = {'image': [], 'labels': []}
    train_dict = {'image': [], 'labels': []}

    # Read the csv file
    csv_file_path = os.path.join(os.getcwd(), "FER Dataset", "icml_face_data.csv")
    with open(csv_file_path, "r") as file:
        
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):                       
            print(f"Processing image {i}", end="\r")
            
            # Convert raw string pixel data into an Image object
            raw_data = [int(pixel) for pixel in row[' pixels'].split()]
            pixel_data = np.array(raw_data).reshape((48, 48))
            image = Image.fromarray(pixel_data.astype('uint8')).convert('RGB')

            # Get images augmented with noise
            images = noise_augmentation(pixel_data)

            # Save data into dictionary depending on whether its labelled testing or training
            if row[' Usage'] == "Testing":
                test_dict['image'].append(image)
                test_dict['image'].extend(images)
                test_dict['labels'].extend([int(row['emotion'])] * 4)
            elif row[' Usage'] == "Training":
                train_dict['image'].append(image)
                train_dict['image'].extend(images)
                train_dict['labels'].extend([int(row['emotion'])] * 4)

    # Split the training data into training + validation with a 90-10 ratio
    combined_data = list(zip(train_dict['image'], train_dict['labels']))
    train_data_combined, validation_data_combined = train_test_split(combined_data, test_size=0.1, random_state=42)
    train_data = {'image': [item[0] for item in train_data_combind], 'labels': [item[1] for item in train_data_combined]}
    validation_data = {'image': [item[0] for item in validation_data_combined], 'labels': [item[1] for item in validation_data_combined]}

    # Construct the Dataset Dictionary
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict(train_data), 
        'test': Dataset.from_dict(test_dict),
        'validation': Dataset.from_dict(validation_data)
    })

    return dataset_dict


def noise_augmentation(pixel_data):
    """
    Add noise to data to make model training more robust
    Referenced from: https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
    :param pixel_data: Image pixels represented as a numpy array
    :return:           List of the input image augmented with noise
    """
    # Convert pixel datatype to a recognisable format by model
    image = pixel_data.astype('uint8')
    
    # Apply Gaussian noise
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)
    aug_image_arr1 = aug.augment_image(image)
    aug_image1 = Image.fromarray(aug_image_arr1.astype('uint8')).convert('RGB')

    # Apply Poisson noise
    aug = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
    aug_image_arr2 = aug.augment_image(image)
    aug_image2 = Image.fromarray(aug_image_arr2.astype('uint8')).convert('RGB')

    # Apply salt and pepper noise
    aug = iaa.SaltAndPepper(p=0.05)
    aug_image_arr3 = aug.augment_image(image)
    aug_image3 = Image.fromarray(aug_image_arr3.astype('uint8')).convert('RGB')

    return [aug_image1, aug_image2, aug_image3]


def transform(batch):
    """
    Takes in a batch of images, performing resizing and normalization on each image
    :param batch: A list of dictionaries of image pixels
    :return:      Data to be used as inputs to the classification model (image + label)
    """
    inputs = processor([x for x in batch['image']], return_tensors='pt')
    inputs['labels'] = batch['labels']
    return inputs


def collate_fn(batch):
    """
    Concatenate image pixels and labels from a list of dictionaries into tensors
    :param batch: A list of dictionaries of image pixels
    :return:      Batch of tensors to be used as inputs
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load("accuracy")
def compute_metrics(p):
    """
    Compute accuracy of the results
    :param p: Numpy array containing the predicted and actual results
    :return:  Accuracy score from the predicted results
    """
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def define_training_arguments():
    """
    Specifies the arguments to be used for training
    :return: Subset of the arguments related to the training loop
    """
    training_args = TrainingArguments(
        output_dir="./my_model_aug",        # Directory where model predictions and checkpoints are saved
        per_device_train_batch_size=16,     # Batch size per GPU core for training
        evaluation_strategy="steps",        # Perform evaluation every eval_steps
        num_train_epochs=4,                 # Pass through the training set 4 times
        fp16=True,                          # 16-bit precision used for training
        save_steps=100,                     # Save two checkpoints every 100 steps
        eval_steps=100,                     # Take 100 steps before performing evaluation
        logging_steps=10,                   # 10 update steps between two logs
        learning_rate=2e-4,                 # Initial learning rate for AdamW optimizer
        save_total_limit=2,                 # Maintain a maximum of 2 checkpoints
        remove_unused_columns=False,        # Keep the image features to create "pixel_values"
        push_to_hub=False,                  # Do not push the model to HuggingFace hub
        report_to='tensorboard',            # Report training metrics and evaluation board to TensorFlow vis tool
        load_best_model_at_end=True,        # Save the best checkpoint after training
        )
    return training_args


def create_trainer(training_args, prepared_dataset):
    """
    Specifies the data and parameters to be used for training
    :return: Trainer class to be used for model training
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["validation"],
        tokenizer=processor,
    )
    return trainer


def perform_training(trainer):
    """
    Train and saves the resultant model
    """
    train_results = trainer.train()
    trainer.save_model("./my_model_aug")
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


def perform_evaluation(trainer, prepared_dataset):
    """
    Evaluate the model on the specified metrics
    """
    metrics = trainer.evaluate(prepared_dataset['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":

    print(f"Step 1 of 3: Reading data from CSV file")
    dataset = read_data()
    
    print(f"Step 2 of 3: Preparing dataset for training")
    prepared_dataset = dataset.with_transform(transform)

    print(f"Step 3 of 3: Training the Model")
    training_args = define_training_arguments()
    trainer = create_trainer(training_args, prepared_dataset)
    perform_training(trainer)
    perform_evaluation(trainer, prepared_dataset)
