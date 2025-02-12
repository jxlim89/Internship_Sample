"""
Author: Daren Tan
Date: February 27, 2024
Description: Python script to finetune the VideoMAE Classifier on AFEW-VA dataset

Code adapted from:
- Author  = Julián Ansaldo
- Title   = Fine-tuning for Video Classification with 🤗 Transformers
- Journal = GitHub notebook
- Date    = April 19, 2023
- url     = https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb
"""

import os
import torch
import numpy as np
import pytorchvideo.data
from evaluate import load
from datasets import DatasetDict
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, Resize
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, RandomShortSideScale, UniformTemporalSubsample

# Load the pretrained processor and model
model_name_or_path = "MCG-NJU/videomae-base"
processor = VideoMAEImageProcessor.from_pretrained(model_name_or_path)
model = VideoMAEForVideoClassification.from_pretrained(model_name_or_path, num_labels=4)


def read_data():
    """
    Read video data from AVI files
    :return: Dataset (in HuggingFace format) of the train, test, and validation data
    """
    # Define variables to be used for applying transformation on datasets
    mean, std, resize_to, num_frames_to_sample, clip_duration = get_transform_variables()
    
    # Root directory where AVI files are separated into train-test-validation sets
    dataset_root_path = os.path.join(os.getcwd(), "AFEW-VA Videos")
    
    # Get the train, test, and validation datasets
    train_dataset = get_train_dataset(mean, std, resize_to, num_frames_to_sample, clip_duration, dataset_root_path)
    val_dataset, test_dataset = get_val_and_test_dataset(mean, std, resize_to, num_frames_to_sample, clip_duration, dataset_root_path)
    
    # Construct the Dataset Dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset, 
        'test': test_dataset,
        'validation': val_dataset
    })

    return dataset_dict


def get_transform_variables():
    """
    Compute variables required for image and video transformation
    :return: mean, standard deviation, resize dimensions, number of frames to sample, duration of clip
    """
    # Get mean and standard deviation for image normalisation
    mean = processor.image_mean
    std = processor.image_std
    
    # Check if "shortest_edge" is specified in processor size
    if "shortest_edge" in processor.size:
        # If specified, set height and width to the same value (square aspect ratio)
        height = width = processor.size["shortest_edge"]
    else:
        # If not specified, set height and width separately
        height = processor.size["height"]
        width = processor.size["width"]
    
    # Create a tuple for resizing the input images
    resize_to = (height, width)

    # Get number of frames to sample and set other video-related variables
    num_frames_to_sample = model.config.num_frames
    sample_rate = 4 # Frame sampling rate
    fps = 25        # Frames per second
    clip_duration = num_frames_to_sample * sample_rate / fps    # Calculate clip duration in seconds

    return mean, std, resize_to, num_frames_to_sample, clip_duration


def get_train_dataset(mean, std, resize_to, num_frames_to_sample, clip_duration, dataset_root_path):
    """
    Extract the training dataset from AVI files and introduce transformations to diversify the samples
    :param mean:                 Mean of the image processor
    :param std:                  Standard deviation of the image processor
    :param resize_to:            Tuple of (height, width) to rescale a frame to
    :param num_frames_to_sample: Number of frames corresponding to model input size
    :param clip_duration:        Length of the video clip
    :param dataset_root_path:    Root directory storing the AVI video clips for all sets
    :return:                     LabeledVideoDataset object for the training dataset
    """
    # Define a sequence of transformations for the training dataset
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),     # Standardises input size for video frames
                        Lambda(lambda x: x / 255.0),                        # Scale pixel value to between 0 and 1
                        Normalize(mean, std),                               # Normalise the pixel values
                        RandomShortSideScale(min_size=256, max_size=320),   # Randomly scale shorter side of video frame
                        RandomCrop(resize_to),                              # Randomly crop video frame to specified size
                        RandomHorizontalFlip(p=0.5),                        # Randomly flip video frame with probability of 0.5
                    ]
                ),
            ),
        ]
    )

    # Define the training dataset
    train_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )
    return train_dataset


def get_val_and_test_dataset(mean, std, resize_to, num_frames_to_sample, clip_duration, dataset_root_path):
    """
    Extract the validation and testing dataset from AVI files and introduce transformations (except cropping and flipping)
    :param mean:                 Mean of the image processor
    :param std:                  Standard deviation of the image processor
    :param resize_to:            Tuple of (height, width) to rescale a frame to
    :param num_frames_to_sample: Number of frames corresponding to model input size
    :param clip_duration:        Length of the video clip
    :param dataset_root_path:    Root directory storing the AVI video clips for all sets
    :return:                     LabeledVideoDataset object for the training dataset
    """    
    # Define a sequence of transformations for the validation and training dataset
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample), # Standardises input size for video frames
                        Lambda(lambda x: x / 255.0),                    # Scale pixel value to between 0 and 1
                        Normalize(mean, std),                           # Normalise the pixel values
                        Resize(resize_to),                              # Resize video frame to specified size
                    ]
                ),
            ),
        ]
    )

    # Define the validation dataset
    val_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    # Define the testing dataset
    test_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    return val_dataset, test_dataset


def collate_fn(batch):
    """
    Concatenate image pixels and labels from a list of dictionaries into tensors 
    used by `Trainer` to prepare data batches
    :param batch: A list of dictionaries of image pixels
    :return:      Batch of tensors to be used as inputs
    """
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack([x["video"].permute(1, 0, 2, 3) for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}


metric = load("accuracy")
def compute_metrics(p):
    """
    Compute accuracy of the results
    :param p: Numpy array containing the predicted and actual results
    :return:  Accuracy score from the predicted results
    """
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def define_training_arguments(train_dataset):
    """
    Specifies the arguments to be used for training
    :return: Subset of the arguments related to the training loop
    """
    training_args = TrainingArguments(
        output_dir="./checkpoint",        # Directory where model predictions and checkpoints are saved
        per_device_train_batch_size=4,     # Batch size per GPU core for training
        evaluation_strategy="steps",        # Perform evaluation every eval_steps
        num_train_epochs=4,                 # Pass through the training set 4 times
        fp16=True,                          # 16-bit precision used for training
        save_steps=1000,                    # Save two checkpoints every 1000 steps
        logging_steps=10,                   # 10 update steps between two logs
        learning_rate=2e-4,                 # Initial learning rate for AdamW optimizer
        save_total_limit=2,                 # Maintain a maximum of 2 checkpoints
        remove_unused_columns=False,        # Keep the image features to create "pixel_values"
        push_to_hub=False,                  # Do not push the model to HuggingFace hub
        report_to='tensorboard',            # Report training metrics and evaluation board to TensorFlow vis tool
        load_best_model_at_end=True,        # Save the best checkpoint after training
        max_steps=(train_dataset.num_videos // 4) * 4,
    )
    return training_args
    

def create_trainer(training_args, dataset):
    """
    Specifies the data and parameters to be used for training
    :return: Trainer class to be used for model training
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
    )
    return trainer    


def perform_training(trainer):
    """
    Train and saves the resultant model
    """
    train_results = trainer.train()
    trainer.save_model("./finetuned_model_1")
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


def perform_evaluation(trainer, prepared_dataset):
    """
    Evaluate the model on the specified metrics
    """
    metrics = trainer.evaluate(prepared_dataset['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    print(f"Step 1 of 2: Reading data from AVI video files")
    dataset = read_data()

    print(f"Step 2 of 2: Training the Model")
    training_args = define_training_arguments(dataset["train"])
    trainer = create_trainer(training_args, dataset)
    perform_training(trainer)
    perform_evaluation(trainer, dataset)

