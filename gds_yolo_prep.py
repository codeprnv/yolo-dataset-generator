import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

# Functions
def enhance_class_name(class_names):
    return [f"all {class_name}s" for class_name in class_names]

def segment(sam_predictor, image, xyxy):
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def list_files_with_extensions(directory, extensions):
    """
    List all files in the given directory with the specified extensions.
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Ensure GPU access
print("Checking GPU availability...")
gpu_available = torch.cuda.is_available()
print("GPU available:", gpu_available)
if not gpu_available:
    print("Please enable GPU support for better performance.")

# Constants
HOME = os.getcwd()
print("HOME:", HOME)

# Prompt the user for class names
def get_class_names():
    while True:
        class_names_input = input("Please enter the class names, separated by commas (e.g., dog, cat, car): ")
        class_names = [name.strip() for name in class_names_input.split(",") if name.strip()]
        if class_names:
            return class_names
        else:
            print("You must enter at least one class name.")

class_names = get_class_names()

# Load models
GROUNDING_DINO_CONFIG_PATH = r"D:\Pranav\Object-Detection\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = r"D:\Pranav\Object-Detection\GroundingDINO\weights\groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = r"D:\Pranav\Object-Detection\GroundingDINO\weights\sam_vit_h_4b8939.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
SAM_ENCODER_VERSION = "vit_h"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Full Dataset Mask Auto Annotation
def full_dataset_annotation(images_directory, classes, box_threshold, text_threshold):
    images = {}
    annotations = {}
    image_paths = list_files_with_extensions(directory=images_directory, extensions=['jpg', 'jpeg', 'png'])

    print(f"Found {len(image_paths)} image(s) in {images_directory}")

    for image_path in tqdm(image_paths):
        print(f"\nProcessing image: {image_path}")
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        if detections is None or len(detections.xyxy) == 0:
            print(f"No detections for image: {image_path}")
            continue

        detections.mask = segment(sam_predictor=sam_predictor, image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyxy=detections.xyxy)
        images[image_name] = image
        annotations[image_name] = detections

    return images, annotations

images, annotations = full_dataset_annotation(os.path.join(HOME, 'Potato'), class_names, 0.35, 0.25)

# Check if images and annotations are generated
print(f"Number of images: {len(images)}")
print(f"Number of annotations: {len(annotations)}")

# Prompt user for the main folder name
main_folder_name = input("Please enter the main folder name for the dataset: ")

# Create directories in YOLO format
def create_yolo_directories(main_folder_name):
    base_path = os.path.join(HOME, main_folder_name)
    for split in ['train', 'test', 'valid']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)
    return base_path

base_path = create_yolo_directories(main_folder_name)

# Save labels in YOLO format
def save_annotations(images, annotations, classes, base_path, split):
    images_path = os.path.join(base_path, split, 'images')
    labels_path = os.path.join(base_path, split, 'labels')

    for image_name, detections in annotations.items():
        # Save image
        image = images[image_name]
        image_output_path = os.path.join(images_path, image_name)
        cv2.imwrite(image_output_path, image)
        print(f"Saved image to {image_output_path}")

        # Save annotations
        annotation_output_path = os.path.join(labels_path, f"{os.path.splitext(image_name)[0]}.txt")
        with open(annotation_output_path, 'w') as f:
            for bbox, confidence, class_id, mask in zip(detections.xyxy, detections.confidence, detections.class_id, detections.mask):
                x_center = (bbox[0] + bbox[2]) / 2 / image.shape[1]
                y_center = (bbox[1] + bbox[3]) / 2 / image.shape[0]
                width = (bbox[2] - bbox[0]) / image.shape[1]
                height = (bbox[3] - bbox[1]) / image.shape[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        print(f"Saved annotations to {annotation_output_path}")

# Split data
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

image_paths = list(images.keys())
np.random.shuffle(image_paths)
train_split = int(len(image_paths) * train_ratio)
valid_split = int(len(image_paths) * (train_ratio + valid_ratio))

train_images = {k: images[k] for k in image_paths[:train_split]}
valid_images = {k: images[k] for k in image_paths[train_split:valid_split]}
test_images = {k: images[k] for k in image_paths[valid_split:]}

train_annotations = {k: annotations[k] for k in image_paths[:train_split]}
valid_annotations = {k: annotations[k] for k in image_paths[train_split:valid_split]}
test_annotations = {k: annotations[k] for k in image_paths[valid_split:]}

print("Saving training annotations...")
save_annotations(train_images, train_annotations, class_names, base_path, 'train')
print("Saving validation annotations...")
save_annotations(valid_images, valid_annotations, class_names, base_path, 'valid')
print("Saving test annotations...")
save_annotations(test_images, test_annotations, class_names, base_path, 'test')

print(f"YOLO dataset has been created in {base_path}")

def create_data_yaml(base_path, class_names):
    data_yaml_path = os.path.join(base_path, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(base_path, 'train/images')}\n")
        f.write(f"val: {os.path.join(base_path, 'valid/images')}\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    print(f"'data.yaml' file created at {data_yaml_path}")

create_data_yaml(base_path, class_names)
