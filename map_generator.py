import os
import pandas as pd
import numpy as np
import cv2
import json


def load_keypoints_from_csv(csv_path):
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


def generate_heatmap(image_shape, keypoints, sigma=3):
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    for x, y in zip(keypoints['x'], keypoints['y']):
        heatmap = cv2.circle(heatmap, (x, y), sigma, (1,), -1)
    return heatmap / np.max(heatmap)  # Normalize to [0, 1]


def generate_vectormap(image_shape, keypoints):
    vectormap_x = np.zeros(image_shape[:2], dtype=np.float32)
    vectormap_y = np.zeros(image_shape[:2], dtype=np.float32)

    # Assuming you have a list of connected joints, for example:
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
            # Fill the vectormap in the vicinity of the line connecting the two keypoints
            cv2.line(vectormap_x, start_point, end_point, unit_vector[0], 1)
            cv2.line(vectormap_y, start_point, end_point, unit_vector[1], 1)

    vectormap = np.stack([vectormap_x, vectormap_y], axis=-1)
    return vectormap


def save_maps(filename, heatmap, vectormap, heatmap_dir, vectormap_dir):
    # Strip the image extension and append .npy
    base_filename = os.path.splitext(filename)[0] + '.npy'
    heatmap_path = os.path.join(heatmap_dir, base_filename)
    vectormap_path = os.path.join(vectormap_dir, base_filename)
    np.save(heatmap_path, heatmap)
    np.save(vectormap_path, vectormap)


def main():
    # Ensure the directories exist
    os.makedirs('data/mappings/heatmaps', exist_ok=True)
    os.makedirs('data/mappings/vectormaps', exist_ok=True)

    csv_path = 'data/mappings/breaststroke.csv'
    keypoints_data = load_keypoints_from_csv(csv_path)

    for filename, keypoints in keypoints_data.items():
        image_path = os.path.join('data/original_frames', filename).replace('\\', '/')
        print(f"Loading image from: {image_path}")  # Add this line
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        heatmap = generate_heatmap(image.shape, keypoints)
        vectormap = generate_vectormap(image.shape, keypoints)
        save_maps(filename, heatmap, vectormap, 'data/mappings/heatmaps', 'data/mappings/vectormaps')


if __name__ == "__main__":
    main()
