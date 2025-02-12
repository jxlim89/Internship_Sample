"""
@article{toisoul2021estimation,
  author  = {Antoine Toisoul and Jean Kossaifi and Adrian Bulat and Georgios Tzimiropoulos and Maja Pantic},
  title   = {Estimation of continuous valence and arousal levels from faces in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  url     = {https://www.nature.com/articles/s42256-020-00280-0}
}
"""

import gc
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from torchvision import transforms

from emonet.models import EmoNet
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC

# Use CUDA-compatible GPU if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def load_model(n_expression):
    """
    Load trained weights into the EmoNet and prepare it for evaluation
    :param n_expression: Number of expressions used when training (5 or 8)
    :return:             EmoNet model loaded with pre-trained weights
    """
    # Load the model weights from a Pytorch state dictionary file (.pth), ensuring compatibility with CPU
    model_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')
    print(f'Loading the model from {model_path}.')
    model_weights = torch.load(str(model_path), map_location='cpu')
    
    # Load pretrained weights into EmoNet model
    model_weights = {k.replace('module.',''):v for k,v in model_weights.items()}
    model = EmoNet(n_expression=n_expression).to(device)
    model.load_state_dict(model_weights, strict=False)

    # Set model to evaluation mode: no gradient computation during inference
    model.eval()

    return model


def get_prediction(clip_path, model):
    """
    For a particular clip, obtain the predicted VA levels for each frame
    :param clip_path: Directory path containing individual frames of the clip
    :param model:     EmoNet model with pre-trained weights
    :return:          Dictionary containing ground truth and predicted VA levels
    """
    # Create the data loaders
    transform_image = transforms.Compose([transforms.ToTensor()])

    # Initialise dictionary to store ground truth and predicted VA levels
    clip_dict = dict()

    with open(clip_path, 'r') as file:
        clip_data = json.load(file)
        clip_name = Path(clip_path).name[:3]

        val_gts = []
        aro_gts = []
        img_list = []

        # Extract ground truth VA levels and image data from each frame
        for frame, frame_data in clip_data[clip_name]["frames"].items():
            
            val_gts.append(frame_data['valence'])
            aro_gts.append(frame_data['arousal'])
            
            # Process image data to correct input format for EmoNet model
            img = np.array(frame_data['image']).reshape((256, 256, 3)).astype(np.uint8)
            img = np.ascontiguousarray(img)
            img = transform_image(img)
            img = img.to(device)
            img_list.append(img)
        
        # Convert list of image tensors into a single tensor to feed into EmoNet model
        images = torch.stack(img_list, dim=0)
        output = model(images)

        # Extract the predicted VA levels
        val_pred = output.get('valence', None).tolist()
        aro_pred = output.get('arousal', None).tolist()

        del output
        torch.cuda.empty_cache()
        gc.collect()

    clip_dict['val_gts'] = val_gts
    clip_dict['aro_gts'] = aro_gts
    clip_dict['val_pred'] = val_pred
    clip_dict['aro_pred'] = aro_pred

    return clip_dict


def process_clips(model, isProcess):
    """
    Use EmoNet model to get predicted VA levels for all AFEW-VA clips
    :param model:     EmoNet model trained on AffectNet images
    :param isProcess: Whether to process the AFEW-VA clips
    """
    # Skip processing if indicated as such
    if not isProcess:
        print("Not reprocessing data. Use --process 1 flag to reprocess")
        return

    ## Get the image path of all files that are to be processed
    json_path = Path.cwd().parent.parent / "Extract AFEW-VA" / "out"
    # json_path = Path.cwd() / "out"
    clip_paths = json_path.glob("*")
    
    # Iterate through all clips to get predicted VA levels
    clips_pred = dict()
    for clip_path in sorted(clip_paths):
        clip_name = Path(clip_path).name
        print(f"Processing {clip_name[:3]} of {600}", end='\r')
        clips_pred[clip_name[:3]] = get_prediction(clip_path, model)

    # Save predicted values into a JSON file
    save_path = Path.cwd() / "pred.json"
    with open(save_path, "w") as file:
        json.dump(clips_pred, file)


def compute_metrics():
    """
    Calculate the CCC score for the AFEW-VA dataset using EmoNet model trained on AffectNet
    """
    # Load the predictions
    data_path = Path.cwd() / "pred.json"
    with open(data_path, 'r') as file:
        VA_data = json.load(file)

    # Extract the VA values
    # win_size = 5
    # assert win_size % 2 == 1
    val_ccc_list, aro_ccc_list = [], []
    val_all_gts, val_all_pred, aro_all_gts, aro_all_pred = [], [], [], []
    for clip in VA_data:    
        
        val_pred = VA_data[clip]['val_pred']
        # padded_val_pred = [val_pred[0]] * (win_size // 2) + val_pred + [val_pred[-1]] * (win_size // 2)
        # val_pred = np.convolve(padded_val_pred, np.ones(win_size) / win_size, mode='valid')
        
        aro_pred = VA_data[clip]['aro_pred']
        # padded_aro_pred = [aro_pred[0]] * (win_size // 2) + aro_pred + [aro_pred[-1]] * (win_size // 2)
        # aro_pred = np.convolve(padded_aro_pred, np.ones(win_size) / win_size, mode='valid')
        
        # Save VA values into a single list
        val_all_gts.append(VA_data[clip]['val_gts'])
        val_all_pred.append(val_pred)
        aro_all_gts.append(VA_data[clip]['aro_gts'])
        aro_all_pred.append(aro_pred)

        # Calculate CCC score for each clip
        val_ccc = CCC(VA_data[clip]['val_gts'], val_pred)
        aro_ccc = CCC(VA_data[clip]['aro_gts'], aro_pred)
        if not np.isnan(val_ccc):
            val_ccc_list.append(val_ccc)
        if not np.isnan(aro_ccc):
            aro_ccc_list.append(aro_ccc)
    
    # Combine all the data
    val_all_gts = np.concatenate(val_all_gts).ravel()
    val_all_pred = np.concatenate(val_all_pred).ravel()
    aro_all_gts = np.concatenate(aro_all_gts).ravel()
    aro_all_pred = np.concatenate(aro_all_pred).ravel()

    # Compute the accuracy score
    cat_gts, cat_pred = [], []
    for val_gts, aro_gts in zip(val_all_gts, aro_all_gts):
        if val_gts <= 0 and aro_gts <= 0:
            cat_gts.append(1)
        elif val_gts <= 0 and aro_gts > 0:
            cat_gts.append(2)
        elif val_gts > 0 and aro_gts <= 0:
            cat_gts.append(3)
        elif val_gts > 0 and aro_gts > 0:
            cat_gts.append(4)
    for val_pred, aro_pred in zip(val_all_pred, aro_all_pred):
        if val_pred <= 0 and aro_pred <= 0:
            cat_pred.append(1)
        elif val_pred <= 0 and aro_pred > 0:
            cat_pred.append(2)
        elif val_pred > 0 and aro_pred <= 0:
            cat_pred.append(3)
        elif val_pred > 0 and aro_pred > 0:
            cat_pred.append(4)
    correct_pred = sum(gts == pred for gts, pred in zip(cat_gts, cat_pred))
    accuracy = correct_pred / len(cat_gts)

    # Compute the CCC score using combined data
    val_all_ccc = CCC(val_all_gts, val_all_pred)
    aro_all_ccc = CCC(aro_all_gts, aro_all_pred)

    # Calculate min, max, mean values for valence and arousal by treating clips separately
    val_min, val_max, val_mean = min(val_ccc_list), max(val_ccc_list), sum(val_ccc_list) / len(val_ccc_list)
    count_val = [sum(-1 <= value < 0.5 for value in val_ccc_list), sum(0.5 <= value <= 1 for value in val_ccc_list)]
    aro_min, aro_max, aro_mean = min(aro_ccc_list), max(aro_ccc_list), sum(aro_ccc_list) / len(aro_ccc_list)
    count_aro = [sum(-1 <= value < 0.5 for value in aro_ccc_list), sum(0.5 <= value <= 1 for value in aro_ccc_list)]

    # Print the results
    print(f"\nCCC score by combining all clips:")
    print(f"Valence: {val_all_ccc:.3f}   Arousal: {aro_all_ccc:.3f}\n")
    print(f"CCC score by considering each clip individually")
    print(f"Valence: ")
    print(f"Min: {val_min:.3f}   Max: {val_max:.3f}   Mean: {val_mean:.3f}")
    print(f"Less than 0.5: {count_val[0]}   More than 0.5: {count_val[1]}")
    print(f"\nArousal: ")
    print(f"Min: {aro_min:.3f}   Max: {aro_max:.3f}   Mean: {aro_mean:.3f}")
    print(f"Less than 0.5: {count_aro[0]}   More than 0.5: {count_aro[1]}\n")
    print(f"Accuracy: {accuracy:.3f}")


def parse_args():
    """
    Function responsible for parsing command-line arguments
    :return: Arguments parsed from command-line
    """
    parser = argparse.ArgumentParser()
    
    # Determine which pretrained weights file to load
    parser.add_argument("--nclass", type=int, default=8, choices=[5,8], help="Number of emotional classes to test the model on. Please use 5 or 8.")
    # Determine whether to run predictions on the test set again
    parser.add_argument("--process", type=int, default=0, choices=[0,1], help="Indicate 0 (no process) or 1 (process).")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.nclass)
    process_clips(model, args.process)
    compute_metrics()
