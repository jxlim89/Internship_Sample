import av
import os
import torch
import numpy as np
import pytorchvideo.data
from train import get_transform_variables
from sklearn.metrics import accuracy_score, f1_score
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, UniformTemporalSubsample

image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEForVideoClassification.from_pretrained("./finetuned_model_1", num_labels=4)

def get_test_dataset(mean, std, resize_to, num_frames_to_sample, clip_duration):
    """
    Extract the validation and testing dataset from AVI files and introduce transformations (except cropping and flipping)
    :param mean:                 Mean of the image processor
    :param std:                  Standard deviation of the image processor
    :param resize_to:            Tuple of (height, width) to rescale a frame to
    :param num_frames_to_sample: Number of frames corresponding to model input size
    :param clip_duration:        Length of the video clip
    :return:                     LabeledVideoDataset object for the test dataset
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

    # Define the testing dataset
    test_dataset = pytorchvideo.data.labeled_video_dataset(
        data_path=os.path.join(os.getcwd(), "AFEW-VA Videos", "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    return test_dataset


def process_data(test_data):

    actual = []
    prediction = []
    total_clips = len(list(test_data))

    for i, data in enumerate(test_data):
        print(f"Processing file {i} of {total_clips}", end="\r")
        
        input = data['video'].permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input)
            logits = outputs.logits
        predicted_label = logits.argmax(-1).item()

        actual.append(data["label"])
        prediction.append(predicted_label)
    
    return prediction, actual


def calculate_metrics(actual, prediction):
    accuracy = accuracy_score(actual, prediction)
    macro_f1 = f1_score(actual, prediction, average="macro")
    print("Accuracy:", accuracy)
    print("Macro F1-score:", macro_f1)

    with open("model_1_metrics.txt", "w") as file:
        file.write(f'Accuracy: {accuracy}\n')
        file.write(f'Macro F1-score: {macro_f1}\n')


def read_video_pyav(file_path):
    '''
    Decode the video with PyAV decoder.

    '''
    # Read in video into a PyAV container and set the file pointer to the start
    container = av.open(file_path)
    container.seek(0)
    
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


if __name__ == "__main__":
    mean, std, resize_to, num_frames_to_sample, clip_duration = get_transform_variables()
    test_data = get_test_dataset(mean, std, resize_to, num_frames_to_sample, clip_duration)
    actual, prediction = process_data(test_data)
    calculate_metrics(actual, prediction)