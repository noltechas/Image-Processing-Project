import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from utils.data_preprocessing import load_keypoints_from_csv
from sklearn.cluster import KMeans


def apply_adaptive_thresholding(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

def apply_color_clustering(image, k=3):
    image_reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k).fit(image_reshaped)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered.reshape(image.shape).astype(np.uint8)
    return clustered_image

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
    # Load original image
    img_original = cv2.imread(f"../data/original_frames/{filename}")

    # Generate thresholded and clustered versions
    img_thresholded = apply_adaptive_thresholding(img_original)
    img_clustered = apply_color_clustering(img_original)

    # Add an additional dimension to the thresholded image only
    img_thresholded = np.expand_dims(img_thresholded, axis=-1)

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

    vector_map = generate_vector_map(img_original.shape[:2], {'x': data['x'], 'y': data['y']})
    images_with_keypoints.append((img_combined, vector_map))


# Separate images and keypoints
X = np.array([item[0] for item in images_with_keypoints])
y = np.array([item[1] for item in images_with_keypoints])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define model for vector maps
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 7)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(2, (1, 1))  # Output 2 channels for x and y vector maps
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
