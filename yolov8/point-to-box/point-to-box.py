import os
import cv2
import json

def point_to_bbox(x, y, box_size=30):
    x1 = int(x - box_size / 2)
    y1 = int(y - box_size / 2)
    x2 = x1 + box_size
    y2 = y1 + box_size
    return x1, y1, x2, y2

def normalize_coordinates(x1, y1, x2, y2, image_width, image_height):
    # Convert coordinates to the center-based YOLO format and normalize
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height

def visualize_bounding_boxes(input_image, json_file, output_image, output_txt):
    # Load image
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Could not read image from {input_image}")
        return
    
    # Get image dimensions
    image_height, image_width, _ = image.shape
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'step_1' in data and 'result' in data['step_1']:
        points = data['step_1']['result']
        
        with open(output_txt, 'w') as fout:
            for point in points:
                try:
                    x = float(point['x'])  # Ensure x and y are floats from JSON
                    y = float(point['y'])
                    x1, y1, x2, y2 = point_to_bbox(x, y)
                    
                    # Normalize coordinates
                    x_center, y_center, width, height = normalize_coordinates(x1, y1, x2, y2, image_width, image_height)
                    
                    # Convert normalized coordinates to absolute pixel values for visualization
                    x1_abs = int((x_center - width / 2) * image_width)
                    y1_abs = int((y_center - height / 2) * image_height)
                    x2_abs = int((x_center + width / 2) * image_width)
                    y2_abs = int((y_center + height / 2) * image_height)
                    
                    # Draw bounding box on the image
                    cv2.rectangle(image, (x1_abs, y1_abs), (x2_abs, y2_abs), (0, 255, 0), 2)
                    
                    # Write normalized bounding box coordinates to output text file
                    line = f"0 {x_center} {y_center} {width} {height}\n"  # Format: object_index x_center y_center width height
                    fout.write(line)
                    
                except Exception as e:
                    print(f"Error processing point {point}: {e}")
    
    # Save annotated image
    cv2.imwrite(output_image, image)
    print(f"Annotated image saved to {output_image}")
    print(f"Normalized bounding boxes saved to {output_txt}")

def process_folders(image_folder, json_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over files in image folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):  # Process only PNG images (adjust as needed)
            input_image = os.path.join(image_folder, filename)
            json_filename = f"{os.path.splitext(filename)[0]}.json"
            input_json = os.path.join(json_folder, json_filename)
            output_image = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_annotated.png")
            output_txt = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            
            visualize_bounding_boxes(input_image, input_json, output_image, output_txt)
            print(f"Processed {filename}")

#CHANGE PATH ACCORDING TO THE DATASET WANTED
image_folder = 'RFCP_ORANGE_RF/val/images'  # Replace with your input folder containing images (*.png)
json_folder = 'RFCP_ORANGE_RF/val/json_labels'  # Replace with your input folder containing JSON files (*.json)
output_folder = 'RFCP_ORANGE_RF/val/valtxt'  # Replace with your output folder where annotated images and txt files will be saved

process_folders(image_folder, json_folder, output_folder)
print("Processing complete.")