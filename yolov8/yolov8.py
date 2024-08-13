import os
import cv2
import numpy as np
import multiprocessing
from collections import Counter
import cvzone
import pandas as pd
from ultralytics import YOLO
import torch

class_list = ["rapessed flower","orange"]

def resize_image(img, target_size):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    new_img[(target_size - nh) // 2: (target_size - nh) // 2 + nh, (target_size - nw) // 2: (target_size - nw) // 2 + nw] = resized_img
    return new_img

def object(img, model, device):
    target_size = 640  # Ensure dimensions are divisible by 32
    resized_img = resize_image(img, target_size)

    img_tensor = torch.from_numpy(resized_img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0  # Normalize image tensor and move to GPU
    results = model(img_tensor)
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")
    object_classes = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        obj_class = class_list[d]
        object_classes.append(obj_class)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(img, f'{obj_class}', (x2, y2), 1, 1)

    return object_classes

def count_objects_in_image(object_classes):
    counter = Counter(object_classes)
    print("Object Count in Image:")
    for obj, count in counter.items():
        print(f"{obj}: {count}")

def test_model_on_images(model, images_folder, device):
    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith(".png")]

    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error reading image: {image_file}")
            continue

        object_classes = object(image, model, device)
        count_text = "Objects Counted:"
        for obj, count in Counter(object_classes).items():
            count_text += f"\n{obj}: {count}"
        cv2.putText(image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        result_image_path = f'RFRB_ORANGE/result2/{os.path.basename(image_file)}'
        cv2.imwrite(result_image_path, image)

def save_model(model, save_path):
    torch.save(model.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path, device):
    model.model.load_state_dict(torch.load(load_path, map_location=device))
    model.model.to(device)
    print("Model loaded successfully")

if __name__ == '__main__':
    multiprocessing.freeze_support()

    device = torch.device('cpu')
    print(f"Using device: {device}")

    model = YOLO("yolov8x.yaml").to(device)  # Ensure model is on GPU
    model.train(data="config.yaml", epochs=30, augment=True, device=device)
    save_model_path = "yolov8x_trained_model.pt"
    
    save_model(model, save_model_path)
    
    # Load the model for testing
    load_model(model, save_model_path, device)
    
    test_path = 'RFRB_ORANGE/test/images'
    test_model_on_images(model, test_path, device)
