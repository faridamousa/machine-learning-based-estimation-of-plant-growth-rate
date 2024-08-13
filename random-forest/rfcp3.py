#FOR RAPESEED FLOWERS ONLY
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import os
import json
import numpy as np

import os
import cv2
import json
import numpy as np

def resize_image(image, target_size):
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def resize_mask(mask, target_size):
    # Resize the mask to the target size
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return resized_mask

def create_binary_mask(center_points, image_shape, radius=10):
    binary_mask = np.zeros(image_shape, dtype=np.uint8)
    
    for center_point in center_points:
        x, y = center_point
        x = int(x)
        y = int(y)
        # Define region around center point
        xmin = max(0, x - radius)
        ymin = max(0, y - radius)
        xmax = min(image_shape[1], x + radius)
        ymax = min(image_shape[0], y + radius)
        # Mark region as foreground
        binary_mask[ymin:ymax, xmin:xmax] = 1
    
    return binary_mask

def convert_annotations(json_folder, image_folder, output_folder, target_size):
    for json_file in os.listdir(json_folder):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(json_folder, json_file)

        # Construct image name from JSON file name
        image_name = os.path.splitext(json_file)[0]  # Remove extension
        image_path = os.path.join(image_folder, image_name)
        
        # Load the image to get its shape
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image '{image_path}'")
            continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            points = data['step_1']['result']
        
        # Convert annotations to binary mask
        binary_mask = create_binary_mask([(point['x'], point['y']) for point in points], img.shape[:2])

        print(f"Unique values in mask for {json_file}:")
        print(np.unique(binary_mask))

        resized_mask = resize_mask(binary_mask, target_size)
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, resized_mask * 255)

# Function to perform superpixel segmentation and save superpixel segmented images
def superpixel_segmentation(image_folder, superpixel_output_folder, target_size, num_segments=1000, compactness=10):
    # Iterate over all images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            # Load the image
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            
            resized_image = resize_image(image, target_size)
            
            # Convert image to CIE L*a*b* color space
            lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)

            # Define parameters for SLIC
            # Adjust these values according to your specific image and requirements
            num_segments = 1000
            compactness = 10

            # Apply SLIC superpixel segmentation
            segments = slic(lab_image, n_segments=num_segments, compactness=compactness)

            # Create an image with superpixel boundaries
            superpixel_boundaries = mark_boundaries(resized_image, segments, color=(0, 255, 0), mode='thick')
            # Save the superpixel segmented image
            superpixel_boundaries = (superpixel_boundaries * 255).astype('uint8')
            output_filename = os.path.splitext(filename)[0] + '_superpixel.png'
            output_path = os.path.join(superpixel_output_folder, output_filename)
            cv2.imwrite(output_path, superpixel_boundaries)

# Define paths for XML annotations, image files, and output masks
json_folder = 'datasets/RFCP/val/labels'
image_folder = 'datasets/RFCP/val/images'
output_folder = 'datasets/RFCP/val/masks'
superpixel_output_folder = 'datasets/RFCP/val/superpixel'

target_size = (430, 560) 
num_segments = 1000
compactness = 10
superpixel_segmentation(image_folder, superpixel_output_folder, target_size, num_segments, compactness)
# Convert annotations to masks
convert_annotations(json_folder, image_folder, output_folder, target_size)


