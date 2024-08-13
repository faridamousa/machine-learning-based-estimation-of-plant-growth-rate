# FOR RAPESEED FLOWERS ONLY 
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

def load_training_data(superpixel_dir, mask_dir):
    images = []
    masks = []

    # Get list of filenames in superpixel directory
    filenames = os.listdir(superpixel_dir)

    # Iterate over filenames
    for filename in filenames:
        # Construct paths for superpixel image and mask
        superpixel_path = os.path.join(superpixel_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace("_superpixel", ""))

        # Load superpixel image and mask
        superpixel_image = cv2.imread(superpixel_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert all values greater than 0 to 1 in the mask image
        mask_image[mask_image > 0] = 1

        if not verify_mask(mask_image):
            continue

        # Append superpixel image and modified mask to respective lists
        images.append(superpixel_image)
        masks.append(mask_image)

    return images, masks

def verify_mask(mask_image):
    unique_values = np.unique(mask_image)
    print("Unique values in mask image:", unique_values)

    # Convert all values greater than 0 to 1
    mask_image[mask_image > 0] = 1

    # Verify if the mask image contains only values 0 and 1
    if len(np.unique(mask_image)) != 2 or not (0 in np.unique(mask_image) and 1 in np.unique(mask_image)):
        print("Error: Mask image does not contain only two values (0 and 1).")
        return False
    else:
        print("Unique values in mask image:", unique_values)
        print("Mask image verification successful.")
        return True

# Function to extract features from an image
def extract_features(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    features = lab_image.reshape((-1, 3))
    print("Extracted features shape:", features.shape)
    print("Extracted features:", features[:10])  # Print first 10 features as an example
    if np.all(features == 0):
        print("Warning: Extracted features are all zeros.")
    return features

# Function to train the classifier
def train_classifier(features, labels):
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    classifier.fit(features, labels)
    return classifier

# Function to apply the classifier to the image and obtain segmented plants
def segment_plants(superpixel_image_path, original_image_path, classifier):
    # Load superpixel image and original image
    superpixel_image = cv2.imread(superpixel_image_path)
    original_image = cv2.imread(original_image_path)

    # Resize original image to 430x560
    original_image_resized = cv2.resize(original_image, (430, 560))

    # Extract features from the superpixel image
    superpixel_features = extract_features(superpixel_image)

    # Predict labels using the classifier
    predicted_labels = classifier.predict(superpixel_features)

    # Reshape predicted labels to match image shape
    predicted_mask = predicted_labels.reshape(superpixel_image.shape[:2])

    # Threshold the predicted mask to obtain binary segmentation
    _, binary_mask = cv2.threshold(predicted_mask, 0.5, 1, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    dilated_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
    
    # Resize the mask to match the original image size
    resized_mask = cv2.resize(dilated_mask, (original_image_resized.shape[1], original_image_resized.shape[0]))

    # Apply the binary mask to the resized original image
    segmented_image = original_image_resized.copy()
    segmented_image[resized_mask == 0] = (0, 0, 0)  # Set non-plant pixels to black
    segmented_image[resized_mask == 1] = (255, 255, 255)  # Set plant pixels to white in orange regions

    # Return both segmented image and predicted mask
    return segmented_image, predicted_mask

def calculate_accuracy(predicted_mask, ground_truth_mask):
    # Flatten the masks
    predicted_mask_flat = predicted_mask.flatten()
    ground_truth_mask_flat = ground_truth_mask.flatten()

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_mask_flat, predicted_mask_flat)

    return accuracy

def calculate_precision_recall_f1(predicted_mask, ground_truth_mask):
    # Flatten the masks
    predicted_mask_flat = predicted_mask.flatten()
    ground_truth_mask_flat = ground_truth_mask.flatten()

    ground_truth_mask_binary = (ground_truth_mask_flat > 0).astype(int)

    # Compute precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(ground_truth_mask_binary, predicted_mask_flat)

    # Calculate F1 score
    f1 = f1_score(ground_truth_mask_binary, predicted_mask_flat)

    return precision, recall, thresholds, f1

# Main function
def main():
    superpixel_dir = "datasets/RFRB/train/superpixel/"
    mask_dir = "datasets/RFRB/train/masks/"
    test_images_dir = "datasets/RFRB/val/superpixel/"
    test_mask_dir = "datasets/RFRB/val/masks/"
    original_dir = "datasets/RFRB/val/images/"
    result_folder = "datasets/RFRB/result_val/"
    # Load training data
    images, masks = load_training_data(superpixel_dir, mask_dir)

    if not images or not masks:
        print("Error: Failed to load training data.")
        return

    # Check if the number of images matches the number of masks
    if len(images) != len(masks):
        print("Error: Number of images does not match number of masks.")
        return

    # Initialize lists to store features and labels
    all_features = []
    all_labels = []

    # Extract features and labels from training data
    for superpixel_image, mask_image in zip(images, masks):
        if superpixel_image is None or mask_image is None:
            print("Error: Failed to load image or mask.")
            continue

        # Extract features from superpixel image
        features = extract_features(superpixel_image)

        # Flatten the mask image to obtain labels
        labels = mask_image.flatten()

        # Append features and labels to the lists
        all_features.append(features)
        all_labels.append(labels)

    # Concatenate features and labels from all images
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    if len(all_features) == 0 or len(all_labels) == 0:
        print("Error: No features or labels extracted.")
        return

    # Train the classifier
    classifier = train_classifier(all_features, all_labels)

    if classifier is None:
        print("Error: Failed to train classifier.")
        return

    accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    
    # Iterate over test images
    for filename in os.listdir(test_images_dir):
        # Load test image
        test_image = cv2.imread(os.path.join(test_images_dir, filename))

        if test_image is None:
            print(f"Error: Failed to load test image {filename}.")
            continue

        print("Loaded test image:", test_image.shape)

        # Extract the corresponding original image path
        original_image_path = os.path.join(original_dir, filename.split('_superpixel')[0] + ".png")

        # Apply the classifier to segment plants
        segmented_plants, predicted_mask = segment_plants(os.path.join(test_images_dir, filename), original_image_path, classifier)

        print("Segmented plants:", segmented_plants.shape)

        # Load the corresponding ground truth mask
        ground_truth_mask_path = os.path.join(test_mask_dir, filename.replace("_superpixel", ""))
        ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth_mask is None:
            print(f"Error: Failed to load ground truth mask {ground_truth_mask_path}.")
            continue

        # Calculate accuracy between predicted mask and ground truth mask
        accuracy = calculate_accuracy(predicted_mask, ground_truth_mask)
        accuracies.append(accuracy)
        print(f"Accuracy for {filename}: {accuracy}")

        precision, recall, _, f1 = calculate_precision_recall_f1(predicted_mask, ground_truth_mask)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        print(f"precision for {filename}: {precision}")
        print(f"recall for {filename}: {recall}")
        print(f"f1 for {filename}: {f1}")

        # Save or display the segmented plants
        output_path = os.path.join(result_folder, f"segmented_plants_{filename}")
        cv2.imwrite(output_path, segmented_plants)

    # Calculate and print the average accuracy
    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy: {average_accuracy}")

    average_f1_score = np.mean(all_f1_scores)
    print(f"Average F1 score: {average_f1_score}")

    average_precision = np.mean(all_precisions)
    print(f"Average precision score: {average_precision}")

    average_recall= np.mean(all_recalls)
    print(f"Average recall score: {average_recall}")

    plt.figure()
    mean_precision = np.mean(all_precisions, axis=0)
    mean_recall = np.mean(all_recalls, axis=0)
    plt.plot(mean_recall, mean_precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
