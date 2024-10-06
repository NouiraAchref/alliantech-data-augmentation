import os
import shutil
import random
from ultralytics import YOLO
import cv2

# Paths
DATASET_PATH = 'augmented_images'  # Your source folder with product images
OUTPUT_DIR = 'yolo_dataset'        # Output directory for YOLOv8 dataset

# YOLOv8 pre-trained model for object detection, running on GPU
model = YOLO('yolo11x.pt').to('cuda')  # Force model to run on GPU

# Create the train/valid/test folder structure
def create_folders():
    for split in ['train', 'valid', 'test']:
        image_folder = os.path.join(OUTPUT_DIR, split, 'images')
        label_folder = os.path.join(OUTPUT_DIR, split, 'labels')

        # Create directories if they don't exist
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(label_folder, exist_ok=True)

# Function to run object detection on an image and generate YOLOv8 labels
# It will label based on the folder (product) name
def detect_and_generate_labels(image_path, label_path, class_id, confidence_threshold=0.1):
    # Load image
    image = cv2.imread(image_path)

    # Resize image to (640, 640) for better detection performance
    image_resized = cv2.resize(image, (640, 640))

    # Run inference using the YOLO model with confidence threshold
    results = model(image_resized, conf=confidence_threshold, iou=0.3)

    # Get the image dimensions (after resizing)
    h, w, _ = image_resized.shape

    # Check if there are detections
    detections = results[0].boxes
    if len(detections) == 0:
        # No detections, return False to skip this image
        print(f"No detections for {image_path}. Skipping.")
        return False

    # Open label file to write the detected bounding boxes in YOLO format
    with open(label_path, 'w') as label_file:
        # Loop through each detection in the result
        for detection in detections:
            # Instead of using detection class_id, we use the folder's class_id
            x_center, y_center, box_width, box_height = detection.xywh[0].cpu().numpy()

            # Normalize the coordinates by image dimensions
            x_center /= w
            y_center /= h
            box_width /= w
            box_height /= h

            # Write the label in YOLO format: class_id x_center y_center width height
            label_file.write(f'{class_id} {x_center} {y_center} {box_width} {box_height}\n')

    # If detections were found, return True to indicate success
    return True

# Function to split images into train, valid, and test sets and generate labels
def split_and_prepare_data():
    # Create folder structure
    create_folders()

    # List all product folders in augmented_images
    product_folders = os.listdir(DATASET_PATH)
    class_names = []
    
    for class_id, product_folder in enumerate(product_folders):
        product_path = os.path.join(DATASET_PATH, product_folder)

        if os.path.isdir(product_path):
            class_names.append(product_folder)  # Add the folder name as a class
            image_files = [f for f in os.listdir(product_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            # Split the images into train (80%), valid (10%), and test (10%) sets
            random.shuffle(image_files)
            num_images = len(image_files)
            train_split = int(0.8 * num_images)
            valid_split = int(0.9 * num_images)

            train_images = image_files[:train_split]
            valid_images = image_files[train_split:valid_split]
            test_images = image_files[valid_split:]

            # Copy and label the images
            copy_and_label_images(train_images, product_path, 'train', class_id)
            copy_and_label_images(valid_images, product_path, 'valid', class_id)
            copy_and_label_images(test_images, product_path, 'test', class_id)

    # After processing, generate the data.yaml file
    generate_yaml_file(class_names)

# Function to copy images and generate YOLO labels for each split (train/valid/test)
def copy_and_label_images(image_files, product_path, split, class_id):
    for img_file in image_files:
        src_img_path = os.path.join(product_path, img_file)
        dst_img_path = os.path.join(OUTPUT_DIR, split, 'images', img_file)
        label_file_name = img_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
        dst_label_path = os.path.join(OUTPUT_DIR, split, 'labels', label_file_name)

        # Detect and generate labels; only proceed if objects are detected
        if detect_and_generate_labels(src_img_path, dst_label_path, class_id):
            # If object detection was successful, copy the image
            shutil.copy(src_img_path, dst_img_path)
        else:
            # If no detection, remove any empty label files (just in case)
            if os.path.exists(dst_label_path):
                os.remove(dst_label_path)

    print(f"Processed {len(image_files)} images for {split} set.")

# Function to generate the data.yaml file
def generate_yaml_file(class_names):
    yaml_content = f"""
train: {os.path.join(OUTPUT_DIR, 'train/images')}
val: {os.path.join(OUTPUT_DIR, 'valid/images')}
test: {os.path.join(OUTPUT_DIR, 'test/images')}

nc: {len(class_names)}  # Number of classes
names: {class_names}     # Class names
"""
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')

    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

    print(f"Generated data.yaml file with {len(class_names)} classes.")

if __name__ == '__main__':
    split_and_prepare_data()  # Start the process
