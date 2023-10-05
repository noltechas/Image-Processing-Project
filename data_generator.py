import os
import cv2
import numpy as np
import pandas as pd
import json
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, csv_path, batch_size=32, shuffle=True):
        self.image_dir = image_dir
        self.keypoints_data = self.load_keypoints_from_csv(csv_path)
        self.filenames = list(self.keypoints_data.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.filenames)

    def load_keypoints_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        keypoints_data = {}
        for _, row in df.iterrows():
            filename = row['filename']

            # Extracting cx and cy from the string
            point_data_str = row['region_shape_attributes']
            point_data = json.loads(point_data_str.replace('""', '"').replace('undefined', '"undefined"'))

            # Check if 'cx' and 'cy' keys are present in point_data
            if 'cx' not in point_data or 'cy' not in point_data:
                continue

            x = point_data['cx']
            y = point_data['cy']

            joint_str = row['region_attributes'].replace('undefined', 'Lhand')  # Handle 'undefined' joint names

            # Check if joint_str is a valid JSON string
            try:
                joint_data = json.loads(joint_str.replace('""', '"'))
            except json.JSONDecodeError:
                continue

            joint_name = joint_data.get('joint', 'Lhand')

            if filename not in keypoints_data:
                keypoints_data[filename] = {'x': [], 'y': [], 'joint': []}
            keypoints_data[filename]['x'].append(x)
            keypoints_data[filename]['y'].append(y)
            keypoints_data[filename]['joint'].append(joint_name)
        return keypoints_data

    def generate_heatmap(self, image_shape, keypoints, sigma=3):
        num_keypoints = 14  # Set this to the expected number of keypoints
        heatmap = np.zeros((*image_shape[:2], num_keypoints), dtype=np.float32)
        for idx, joint in enumerate(sorted(list(set(keypoints['joint'])))):  # Sort the keypoints for consistency
            temp_heatmap = np.zeros(image_shape[:2], dtype=np.float32)  # Temporary 2D array
            for x, y, j in zip(keypoints['x'], keypoints['y'], keypoints['joint']):
                if j == joint:
                    temp_heatmap = cv2.circle(temp_heatmap, (x, y), sigma, (1,), -1)
            heatmap[:, :, idx] = temp_heatmap

            # Check if the maximum value is not zero before normalizing
            max_val = np.max(heatmap[:, :, idx])
            if max_val > 0:
                heatmap[:, :, idx] /= max_val  # Normalize to [0, 1]
        return heatmap


    def generate_vectormap(self, image_shape, keypoints):
        num_keypoints = 14
        vectormap = np.zeros((*image_shape[:2], 2*num_keypoints), dtype=np.float32)

        connections = [('Lshoulder', 'Lelbow'), ('Lhand','Lelbow'), ('Rshoulder', 'Relbow'), ('Rhand', 'Relbow'), ('Rshoulder', 'Lshoulder'),
                       ('Lshoulder', 'Lhip'), ('Rshoulder', 'Rhip'), ('Rhip', 'Lhip'), ('Rhip', 'Rknee'), ('Lhip', 'Lknee'), ('Rknee', 'Rfoot'),
                       ('Rknee', 'Rfoot'), ('Rshoulder', 'head'), ('Lshoulder', 'head'),]  # Add more connections as needed


        for start, end in connections:
            if start in keypoints['joint'] and end in keypoints['joint']:
                start_idx = keypoints['joint'].index(start)
                end_idx = keypoints['joint'].index(end)
                start_point = (keypoints['x'][start_idx], keypoints['y'][start_idx])
                end_point = (keypoints['x'][end_idx], keypoints['y'][end_idx])
                direction = np.array(end_point) - np.array(start_point)
                unit_vector = direction / np.linalg.norm(direction)

                # Determine the channels for the vectormap
                start_channel = sorted(list(set(keypoints['joint']))).index(start) * 2
                end_channel = sorted(list(set(keypoints['joint']))).index(end) * 2

                # Create temporary 2D arrays for the x and y components
                temp_map_x = np.zeros(image_shape[:2], dtype=np.float32)
                temp_map_y = np.zeros(image_shape[:2], dtype=np.float32)

                # Draw the lines on the temporary 2D arrays
                cv2.line(temp_map_x, start_point, end_point, unit_vector[0], 1)
                cv2.line(temp_map_y, start_point, end_point, unit_vector[1], 1)

                # Assign the temporary 2D arrays to the appropriate channels of the vectormap
                vectormap[:, :, start_channel] = temp_map_x
                vectormap[:, :, start_channel+1] = temp_map_y

        vectormap = cv2.resize(vectormap, (image_shape[1], image_shape[0]))  # Resize to match the input shape
        return vectormap



    def display_image_with_keypoints(self, image, keypoints):
        """Displays the image with keypoints overlaid."""
        img_copy = image.copy()
        for x, y in zip(keypoints['x'], keypoints['y']):
            cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)  # Green color for keypoints
        cv2.imshow('Image with Keypoints', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = []
        batch_heatmaps = []
        batch_vectormaps = []

        for filename in batch_filenames:
            image_path = os.path.join(self.image_dir, filename)
            image = cv2.imread(image_path)

            # Scale down the image to half its original size
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

            keypoints = self.keypoints_data[filename]
            heatmap = self.generate_heatmap(image.shape, keypoints)
            vectormap = self.generate_vectormap(image.shape, keypoints)

            # Resize the image, heatmap, and vectormap to a consistent size if necessary
            target_shape = (320, 240)  # Adjust this to half of your previous desired shape
            image = cv2.resize(image, target_shape)
            heatmap = cv2.resize(heatmap, target_shape)
            vectormap = cv2.resize(vectormap, target_shape)

            batch_images.append(image)
            batch_heatmaps.append(heatmap)
            batch_vectormaps.append(vectormap)

        return np.array(batch_images), [np.array(batch_heatmaps), np.array(batch_vectormaps)]


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filenames)
