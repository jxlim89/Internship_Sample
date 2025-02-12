import os
import cv2
import glob
import time
import face_detection


# Initialise the face detector using Dual Shot Face Detector (DSFD)
face_detector = face_detection.build_detector("DSFDDetector", max_resolution=1080)

def face_crop(img):
    """
    Crop faces from image, and draw a bounding box over the original image
    :param img: Image to be processed for facial detection
    :return: List containing cropped faces
    """
    # Run the face detector on the image
    faces = face_detector.detect(img)

    crop_faces = []
    for face in faces:
        confidence = face[4]
        if confidence > 0.90:
            xmin, ymin, xmax, ymax = [int(_) for _ in face[:4]]
            
            # Save the crop image into a list
            crop_faces.append(img[ymin:ymax, xmin:xmax])
            
            # Draw a bounding box over the original image
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    return crop_faces


def image_processing(img_dir):
    """
    Read images from directory and identify faces from them
    Also crops out the faces and save them into a JSON file
    :param img_dir: Directory of images to be processed
    :return: Cropped and resized images in a list
    """
    # Create (if doesn't exist) the save directory
    sav_dir = os.path.join(os.getcwd(), "Image_to_Face")
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)
    
    # Search path for .jpg and .png files to process
    img_path_list = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

    resized_imgs = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        crop_faces = face_crop(img)

        # Save original images with boundary boxes around faces
        img_name = os.path.basename(img_path).split(".")[0]
        save_path = os.path.join(sav_dir, f"{img_name}_out.jpg")
        cv2.imwrite(save_path, img)
        
        # Save cropped out and resized faces
        for i, crop_face in enumerate(crop_faces):
            resized_img = cv2.resize(crop_face, (96, 96))
            resized_imgs.append(resized_img)
            
            # save_path = os.path.join(sav_dir, f"{img_name}_crop{i}.jpg")
            # cv2.imwrite(save_path, resized_img)

    # Display statistics on image processing
    print(f"{len(img_path_list)} images processed, {len(resized_imgs)} faces detected.")
    
    return resized_imgs
    

def video_processing():
    pass


if __name__ == "__main__":
    img_path = r"C:\Users\tjunheng\Desktop\DSO_Internship_JanApr24\Week 2 (Preparing the Dataset)\Extract Face (Dumb)\Test Images"
    image_processing(img_path)