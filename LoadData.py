import cv2
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt

# Directory containing feature images
features_directory = './data/data/'
# CSV file with labels and image paths
labels_file = './data/data/driving_log.csv'

def preprocess(img):
    """
    Preprocesses an image by converting it to HSV, extracting the S channel,
    and resizing it to 40x40.

    Parameters:
        img (numpy array): Input RGB image.

    Returns:
        resized (numpy array): Processed image (40x40).
    """
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    return resized

def data_loading(delta):
    """
    Loads image paths and steering angle labels from a CSV file, preprocesses the images,
    and applies adjustments to the labels for side camera images.

    Parameters:
        delta (float): Adjustment factor for side camera labels.

    Returns:
        features (list): Preprocessed image features.
        labels (list): Steering angle labels.
    """
    logs = []
    features = []
    labels = []

    # Read CSV file and discard the header row
    with open(labels_file, 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            logs.append(line)
        logs.pop(0)

    # Process each log entry
    for i in range(len(logs)):
        for j in range(3):  # Iterate through center, left, and right image paths
            img_path = logs[i][j]
            img_path = features_directory + 'IMG' + (img_path.split('IMG')[1]).strip()
            img = plt.imread(img_path)  # Read the image
            features.append(preprocess(img))  # Preprocess the image

            # Adjust the labels based on the image type (center, left, right)
            if j == 0:
                labels.append(float(logs[i][3]))  # Center image label
            elif j == 1:
                labels.append(float(logs[i][3]) + delta)  # Left image label
            else:
                labels.append(float(logs[i][3]) - delta)  # Right image label

    return features, labels

# Define adjustment factor for side images
delta = 0.2
# Load and preprocess data
features, labels = data_loading(delta)

# Convert features and labels to numpy arrays
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')

# Save the features and labels to pickle files
with open("features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)