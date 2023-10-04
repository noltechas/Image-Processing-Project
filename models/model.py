import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from utils.data_preprocessing import load_keypoints_from_csv

# Load keypoints
csv_path = '../data/mappings/breaststroke.csv'
keypoints_data = load_keypoints_from_csv(csv_path)

# Load images and prepare data
images = []
keypoints = []

# Determine the expected number of keypoints
expected_keypoints = max(len(data['x']) for data in keypoints_data.values())

images_with_keypoints = []

for filename, data in keypoints_data.items():
    # Load original, thresholded, and clustered images
    img_original = cv2.imread(f"../data/original_frames/{filename}")
    img_thresholded = cv2.imread(f"../data/thresholded_frames/{filename}")
    img_clustered = cv2.imread(f"../data/clustered_frames/{filename}")

    # Normalize and concatenate images along the channel dimension
    img_combined = np.concatenate([
        img_original / 255.0,
        img_thresholded / 255.0,
        img_clustered / 255.0
    ], axis=-1)

    images.append(img_combined)

    # Serialize keypoints in a consistent order for each image
    kp_x = np.array(data['x']) / 480
    kp_y = np.array(data['y']) / 640

    # Fill in missing keypoints with default value
    while len(kp_x) < expected_keypoints:
        kp_x = np.append(kp_x, -1)
        kp_y = np.append(kp_y, -1)

    serialized_keypoints = np.hstack((kp_x, kp_y))  # Serialize as [x1, x2, ..., xn, y1, y2, ..., yn]
    images_with_keypoints.append((img_combined, serialized_keypoints))

# Separate images and keypoints
X = np.array([item[0] for item in images_with_keypoints])
y = np.array([item[1] for item in images_with_keypoints])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 9)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
