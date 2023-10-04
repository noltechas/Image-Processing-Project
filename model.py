import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from utils.data_preprocessing import load_keypoints_from_csv
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def apply_adaptive_thresholding(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

def generate_vector_map(image_shape, keypoints):
    h, w = image_shape[0], image_shape[1]
    vector_map = np.zeros((h, w, 2))

    for i in range(h):
        for j in range(w):
            min_distance = float('inf')
            closest_keypoint = None

            for kp_x, kp_y in zip(keypoints['x'], keypoints['y']):
                distance = np.sqrt((kp_x - j)**2 + (kp_y - i)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_keypoint = (kp_x, kp_y)

            vector_map[i, j, 0] = closest_keypoint[0] - j
            vector_map[i, j, 1] = closest_keypoint[1] - i

    return vector_map

def normalize_vector_map(vector_map):
    # Shift all values to be positive
    shifted_vector_map = vector_map - np.min(vector_map)
    # Normalize to [0, 1]
    normalized_vector_map = shifted_vector_map / np.max(shifted_vector_map)
    return normalized_vector_map

# Load keypoints
csv_path = 'data/mappings/breaststroke.csv'
keypoints_data = load_keypoints_from_csv(csv_path)

# Load images and prepare data
images = []
vector_maps = []

for filename, data in keypoints_data.items():
    # Load original image and resize it
    img_original = cv2.imread(f"data/original_frames/{filename}")
    img_resized = cv2.resize(img_original, (240, 320))  # Half the original size

    # Generate thresholded version
    img_thresholded = apply_adaptive_thresholding(img_resized)

    # Add an additional dimension to the thresholded image
    img_thresholded = np.expand_dims(img_thresholded, axis=-1) / 255.0
    images.append(img_thresholded)

    # Generate and normalize vector map
    vm = generate_vector_map(img_resized.shape[:2], {'x': data['x'], 'y': data['y']})
    normalized_vm = normalize_vector_map(vm)
    vector_maps.append(normalized_vm)

# Separate images and keypoints
X = np.array(images)
y = np.array(vector_maps)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define model for vector maps
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(320, 240, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),  # Upsample the feature maps
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),  # Upsample the feature maps
    Conv2D(2, (1, 1), activation='sigmoid')  # Output 2 channels for x and y vector maps
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2500)

# Save the model
model_path = "my_model.h5"
model.save(model_path)

print(f"Model saved to {model_path}")
