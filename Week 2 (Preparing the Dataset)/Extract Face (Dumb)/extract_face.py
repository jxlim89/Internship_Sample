import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image
from datetime import datetime


# Establish detectors
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def get_dir_list(dir):
    """
    Obtain a list of all the filepaths in a directory
    :param directory: Directory where the files are stored
    :return: List containing filepaths in directory
    """
    img_dir_list = []
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path):
            img_dir_list.append(file_path)
    return img_dir_list


def crop_face(img):
    """
    Extracts square portrait of individual from a frame
    :param img: Numpy array containing RGB pixel value of image
    :return: List containing facial images as numpy array
             Each list element is a tuple containing normal and grayscale cropped images
    """
    # Load some pre-trained data on face frontal from opencv (haar cascade algorithm)
    faces = face_detector.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=[50, 50])

    # Crop faces from images
    cropped_imgs = []
    for (x, y, w, h) in faces:
        cropped_img = img[int(y):int(y + h), int(x):int(x + w)]
        cropped_imgs.append((cropped_img, cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)))

    return cropped_imgs


def get_dist_btw_pts(a, b):
    """
    Calculates distance between 2 points: d = √((x_a - x_b)² + (y_a - y_b)²)
    :param a: The first point
    :param b: The second point
    :return: Distance between the 2 cartesian points
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def align_face(cropped_imgs):
    """
    Rotates the image such that the eyes are level (assuming eyes are detected)
    :param cropped_imgs: List containing cropped images in both normal and grayscale
    :return: List containing aligned images (and original cropped images for those with poor eye detection)
    """
    new_img_list = []
    for (img, gray_img) in cropped_imgs:
        
        # Determine if less/more than 2 eyes are detected, then run a looser check to see if face exists
        # If face exists, add the cropped images (w/o rotation) to list
        eyes = eye_detector.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=3, maxSize=[20, 20])
        if len(eyes) != 2:
            faces = face_detector.detectMultiScale(img)
            if len(faces) == 1:
                new_img_list.append(img)
            continue

        # Assigning left and right eye based on x-value
        if eyes[0][0] > eyes[1][0]:
            left_eye, right_eye = eyes[1], eyes[0]
        else:
            left_eye, right_eye = eyes[0], eyes[1]

        # Obtain center of left and right eye
        right_eye_top_center_x = int(right_eye[0] + (right_eye[2]/2))
        right_eye_top_center_y = int(right_eye[1] + (right_eye[3]/2))
        right_eye_top_center = (right_eye_top_center_x, right_eye_top_center_y)
        left_eye_top_center_x = int(left_eye[0] + (left_eye[2] / 2))
        left_eye_top_center_y = int(left_eye[1] + (left_eye[3] / 2))
        left_eye_top_center = (left_eye_top_center_x, left_eye_top_center_y)

        # Finding rotation direction
        if left_eye_top_center_y > right_eye_top_center_y:
            point_3rd = (right_eye_top_center_x, left_eye_top_center_y)
            direction = -1  # Clockwise rotation
        else:
            point_3rd = (left_eye_top_center_x, right_eye_top_center_y)
            direction = 1  # Anti-clockwise rotation

        # Get angle of rotation
        a = get_dist_btw_pts(left_eye_top_center, point_3rd)
        b = get_dist_btw_pts(right_eye_top_center, point_3rd)
        c = get_dist_btw_pts(right_eye_top_center, left_eye_top_center)
        cos_a = (b*b + c*c - a*a) / (2*b*c)
        angle = (np.arccos(cos_a) * 180) / math.pi
        if direction == -1:
            angle = 90 - angle
        else:
            angle = angle - 90

        # Rotate image
        new_img = Image.fromarray(img)
        aligned_img = np.array(new_img.rotate(direction * angle))
        new_img_list.append(aligned_img)
    
    return new_img_list


def image_processing(img_dir):
    """
    Extract and save faces (if any) from all images in a directory
    :param sav_dir: Directory to save the extracted images 
    """
    # Create (if doesn't exist) the save directory
    dt_str = datetime.now().strftime("%y%m%d_%H%M%S")
    sav_dir = os.path.join(os.getcwd(), "Image_to_Face", dt_str)
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)
    
    img_dir_list = get_dir_list(img_dir)
    
    for img_cnt, img_dir in enumerate(img_dir_list):
        cropped_imgs = crop_face(cv2.imread(img_dir))
        img_list = align_face(cropped_imgs)
        
        # Resize image for consistency and save them into the directory
        for sub_img_cnt, img in enumerate(img_list):
            save_path = os.path.join(sav_dir, f"{img_cnt}_{sub_img_cnt}.png")
            resized_img = cv2.resize(img, (100, 100))
            cv2.imwrite(save_path, resized_img)
    
    print(f"{img_cnt + 1} images processed. {len(os.listdir(sav_dir))} images extracted.")
    print("Images are labelled '<number1>_<number2>.png'\n" + 
          "- <number1> is the number of the original image\n" +
          "- <number2> is the face number of the original image")


def video_processing(vid_dir):
    """
    
    """
    dt_str = datetime.now().strftime("%y%m%d_%H%M%S")
    vid_dir_list = get_dir_list(vid_dir)

    for vid_ct, vid_dir in enumerate(vid_dir_list):
        print(f"Processing video {vid_ct}", end="\r")
        
        # Extract all frames from video capture
        frames = []
        cap = cv2.VideoCapture(vid_dir)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Create (if doesn't exist) the save directory
        sav_dir = os.path.join(os.getcwd(), "Video_to_Face", dt_str, str(vid_ct))
        if not os.path.exists(sav_dir):
            os.makedirs(sav_dir)

        # Crop and align image
        for fr_ct, frame in enumerate(frames):
            cropped_imgs = crop_face(frame)
            img_list = align_face(cropped_imgs)

            # Resize and save the image
            for img_ct, img in enumerate(img_list):
                save_path = os.path.join(sav_dir, f"{fr_ct}_{img_ct}.png")
                resized_img = cv2.resize(img, (100, 100))
                cv2.imwrite(save_path, resized_img)



def handle_command_input():
    """
    Handle the command line input to the Python program
    :return: Mode: 0 for image processing; 1 for video processing
             Directory: Location where images are to be saved
    """
    # Define command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', type=int, choices=[0, 1], required=True, help="Specify the processing mode:\n" +
                                                                            "0 - Perform image processing\n" +
                                                                            "1 - Perform video processing")
    parser.add_argument('-d', type=str, required=True, help="Specify an path to the directory containing files for processing.")

    # Parse command-line arguments
    args = parser.parse_args()
    return args.m, args.d


if __name__=="__main__": 
    mode, dir = handle_command_input()
    if mode == 0:
        image_processing(dir)
    elif mode == 1:
        video_processing(dir)