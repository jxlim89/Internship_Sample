import os
import glob
import json
import face_alignment
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def get_directory():
    """
    Locate the directory where images are saved
    :return: Image folder path
    """
    parent_dir = os.path.dirname(os.getcwd())
    img_dir = os.path.join(parent_dir, "Extract Face (Dumb)", "Test Images")
    return img_dir


def get_images(dir):
    """
    Get the filepaths of all images in a given directory
    :param dir: Folder where images are stored
    :return: List of image filepaths
    """
    img_paths = glob.glob(os.path.join(dir, '*.jpg')) + \
                glob.glob(os.path.join(dir, '*.jpeg')) + \
                glob.glob(os.path.join(dir, '*.png'))

    img_list = []
    for img_path in sorted(img_paths):
        img = io.imread(img_path)
        img_list.append(img)
    
    return img_list


def create_save_directory():
    """
    Creats directory to save the 2d and 3d facial landmark output
    :return: Name of save directories
    """
    # Create the save directory for 2d facial landmarks (if doesn't exist)
    twod_sav_dir = os.path.join(os.getcwd(), "2d_out")
    if not os.path.exists(twod_sav_dir):
        os.makedirs(twod_sav_dir)
    
    # Create the save directory for 3d facial landmarks (if doesn't exist)
    threed_sav_dir = os.path.join(os.getcwd(), "3d_out")
    if not os.path.exists(threed_sav_dir):
        os.makedirs(threed_sav_dir)
    
    return twod_sav_dir, threed_sav_dir


def plot_lines(ax, start_index, end_index, x_coords, y_coords, z_coords):
    """
    Connect adjacent coordinates in a given range
    :param ax:          axes object containing the coordinate plot
    :param start_index: Index of first coordinate in the range
    :param end_index:   Index of second last coordinate in the range
    :param x_coords:    x-axis coordinates
    :param y_coords:    y-axis coordinates
    :param z_coords:    z-axis coordinates (if any)
    """
    if z_coords == []:
        for i in range(start_index, end_index):
            ax.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], color='red')
    else:
        for i in range(start_index, end_index):
            ax.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], [z_coords[i], z_coords[i + 1]], color='red')
    

def connect_lines(ax, x_coords, y_coords, z_coords=[]):
    """
    Given a set of coordinates for facial features, connect the appropriate coordinates to form lines
    :param ax:       axes object containing the coordinate plot
    :param x_coords: x-axis coordinates
    :param y_coords: y-axis coordinates
    :param z_coords: z-axis coordinates (if any)
    """
    plot_lines(ax, 0, 16, x_coords, y_coords, z_coords)   # Ear -> Chin -> Ear
    plot_lines(ax, 17, 21, x_coords, y_coords, z_coords)  # Left eyebrow
    plot_lines(ax, 22, 26, x_coords, y_coords, z_coords)  # Right eyebrow
    plot_lines(ax, 27, 30, x_coords, y_coords, z_coords)  # Nose
    plot_lines(ax, 31, 35, x_coords, y_coords, z_coords)  # Philtrum (area below nose)
    plot_lines(ax, 36, 41, x_coords, y_coords, z_coords)  # Left eye
    plot_lines(ax, 42, 47, x_coords, y_coords, z_coords)  # Right eye
    plot_lines(ax, 48, 67, x_coords, y_coords, z_coords)  # Mouth

def extract_2d_landmarks(img_list, sav2d):
    """
    Extract 2d facial landmarks from images (if any)
    :param img_list: List containing pixel data of images
    :param sav2d:    Save directory
    """
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    
    landmarks = dict()
    for i, img in enumerate(img_list):
        # Obtain the predicted 2d landmarks for each face in image
        preds = fa.get_landmarks(img)

        # If no face detected, skip image
        if preds is None:
            continue
        
        # Solve issue with "preds" treated as ndarray instead of list
        landmarks[str(i + 1)] = np.array(preds).tolist()

        # Plot landmarks on top of image
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for pred in preds:
            x_coords = [x for x, y in pred]
            y_coords = [y for x, y in pred]
            
            for x, y in pred:
                ax.plot(x, y, marker='o', markersize=2, color='red')
            connect_lines(ax, x_coords, y_coords)

        # Save the image with landmarks
        sav_path = os.path.join(sav2d, f"{i + 1}_out.png")
        plt.axis("off")
        plt.savefig(sav_path)
        
        # Close plot to save memory
        plt.cla()
        plt.close(fig)

    # Save landmarks in a JSON file
    with open(os.path.join("2d_out", "2d_out.json"), 'w') as json_file:
        json.dump(landmarks, json_file)


def extract_3d_landmarks(img_list, sav3d):
    """
    Extract 3d facial landmarks from images (if any)
    :param img_list: List containing pixel data of images
    :param sav2d:    Save directory
    """
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    landmarks  = dict()
    for i, img in enumerate(img_list):
        # Obtain the predicted 3d landmarks for each face in image
        preds = fa.get_landmarks(img)
        
        # If no face detected, skip image
        if preds is None:
            continue

        # Solve issue with "preds" treated as ndarray instead of list
        landmarks[str(i + 1)] = np.array(preds).tolist()

        # Plot each cluster of facial landmarks on its own canvas
        fig = plt.figure()
        for j, pred in enumerate(preds):
            x_coords = [x for x, y, z in pred]
            y_coords = [y for x, y, z in pred]
            z_coords = [z for x, y, z in pred]
            
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_coords, y_coords, z_coords, color='red', marker='o', s=10)
            connect_lines(ax, x_coords, y_coords, z_coords)
            
            sav_path = os.path.join(sav3d, f"{i + 1}_{j + 1}_out.png")
            plt.savefig(sav_path)
        plt.cla()
        plt.close(fig)
        
    # Save landmarks in a JSON file
    with open(os.path.join("3d_out", "3d_out.json"), 'w') as json_file:
        json.dump(landmarks, json_file)


if __name__ == "__main__":
    dir = get_directory()
    images = get_images(dir)
    sav2d, sav3d = create_save_directory()
    extract_2d_landmarks(images, sav2d)
    #extract_3d_landmarks(images, sav3d)