import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_images(folder):
    images = []
    labels = []
    for label in ['cat', 'dog']:
        path = os.path.join(folder, label)
        if not os.path.exists(path):
            print(f"Directory does not exist: {path}")
            continue
        print(f"Loading images from: {path}")
        count = 0
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img = cv2.resize(img, (64, 64))  # Resize to 64x64 for simplicity
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            images.append(img)
            labels.append(0 if label == 'cat' else 1)
            count += 1
        print(f"Loaded {count} images from {label} directory.")
    return np.array(images), np.array(labels)

# Correct the path to match your directory structure
train_images, train_labels = load_images(r"C:\Users\Kavya Bhatt\Downloads\train")
test_images, test_labels = load_images(r"C:\Users\Kavya Bhatt\Downloads\test1")

# Check if images and labels are loaded correctly
if train_images.size == 0 or test_images.size == 0:
    print("No images loaded. Please check the directory paths and structure.")
    exit()

# Print the unique labels and their counts
unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)
print(f"Train labels distribution: {dict(zip(unique_train_labels, train_counts))}")
print(f"Test labels distribution: {dict(zip(unique_test_labels, test_counts))}")

if len(unique_train_labels) < 2:
    print("Training data does not have at least two classes. Please ensure the training data contains images of both cats and dogs.")
    exit()

# Flatten and standardize the images
train_images_flat = train_images.reshape(len(train_images), -1)
test_images_flat = test_images.reshape(len(test_images), -1)

scaler = StandardScaler()
train_images_flat = scaler.fit_transform(train_images_flat)
test_images_flat = scaler.transform(test_images_flat)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(train_images_flat, train_labels)

# Evaluate the model
test_predictions = svm.predict(test_images_flat)
accuracy = accuracy_score(test_labels, test_predictions)

print(f'Accuracy: {accuracy * 100:.2f}%')

