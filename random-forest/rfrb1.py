# FOR ORANGES ONLY
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import os
import xml.etree.ElementTree as ET
import numpy as np

def convert_annotations(xml_folder, image_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_folder, xml_file)
        image_name = os.path.splitext(xml_file)[0] + '.png'
        image_path = os.path.join(image_folder, image_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Load the image to get its shape
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image '{image_path}'")
            continue
        
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != 'orange':  # Skip if not the desired class
                continue
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            # Ensure bounding box coordinates are integers
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            # Set the region inside the bounding box to 1 (foreground)
            mask[ymin:ymax, xmin:xmax] = 1  # Set to 1 (white) for foreground
        print(f"Unique values in mask for {xml_file}:")
        print(np.unique(mask))
        output_path = os.path.join(output_folder, os.path.splitext(xml_file)[0] + '.png')
        cv2.imwrite(output_path, mask * 255)  # Save as 0 and 255 (black and white)

# Function to perform superpixel segmentation and save superpixel segmented images
def superpixel_segmentation(image_folder, superpixel_output_folder, num_segments=1000, compactness=10):
    # Iterate over all images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            # Load the image
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            # Convert image to CIE L*a*b* color space
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Define parameters for SLIC
            # Adjust these values according to your specific image and requirements
            num_segments = 1000
            compactness = 10

            # Apply SLIC superpixel segmentation
            segments = slic(lab_image, n_segments=num_segments, compactness=compactness)

            # Create an image with superpixel boundaries
            superpixel_boundaries = mark_boundaries(image, segments, color=(0, 255, 0), mode='thick')
            # Save the superpixel segmented image
            superpixel_boundaries = (superpixel_boundaries * 255).astype('uint8')
            output_filename = os.path.splitext(filename)[0] + '_superpixel.png'
            output_path = os.path.join(superpixel_output_folder, output_filename)
            cv2.imwrite(output_path, superpixel_boundaries)

# Define paths for XML annotations, image files, and output masks
xml_folder = 'datasets/RFRB_ORANGE/val/labels'
image_folder = 'datasets/RFRB_ORANGE/val/images'
output_folder = 'datasets/RFRB_ORANGE/val/masks'
superpixel_output_folder = 'datasets/RFRB_ORANGE/val/superpixel'

num_segments = 1000
compactness = 10
superpixel_segmentation(image_folder, superpixel_output_folder, num_segments, compactness)
# Convert annotations to masks
convert_annotations(xml_folder, image_folder, output_folder)


