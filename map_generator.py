import numpy as np
import cv2, csv

def generate_heatmap(height, width, keypoints, sigma=2):
    heatmap = np.zeros((height, width, len(keypoints)), dtype=np.float32)
    
    for i, (x, y) in enumerate(keypoints):
        heatmap[:, :, i] = cv2.GaussianBlur((heatmap[:, :, i] + draw_gaussian(heatmap[:, :, i], (x, y), sigma)), (0, 0), sigma)
    
    return heatmap

def draw_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    ul = [int(center[0] - tmp_size), int(center[1] - tmp_size)]
    br = [int(center[0] + tmp_size + 1), int(center[1] + tmp_size + 1)]
    if ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
    img_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
    img_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return heatmap

def generate_vectormap(height, width, keypoints_pairs):
    vectormap = np.zeros((height, width, len(keypoints_pairs)*2), dtype=np.float32)
    
    for i, (kp1, kp2) in enumerate(keypoints_pairs):
        vectormap[:, :, i*2:i*2+2] = draw_vector(vectormap[:, :, i*2:i*2+2], kp1, kp2)
    
    return vectormap

def draw_vector(vectormap, kp1, kp2, thickness=1):
    # Calculate unit vector between kp1 and kp2
    length = np.linalg.norm(np.array(kp2) - np.array(kp1))
    if length == 0:
        return vectormap
    dx = (kp2[0] - kp1[0]) / length
    dy = (kp2[1] - kp1[1]) / length

    # Draw line on the vectormap
    cv2.line(vectormap[:,:,0], tuple(kp1), tuple(kp2), dx, thickness)
    cv2.line(vectormap[:,:,1], tuple(kp1), tuple(kp2), dy, thickness)
    
    return vectormap

# 1. Extract Keypoints
def extract_keypoints_from_csv(csv_path):
    keypoints = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            cx = int(row[5].split('"cx":')[1].split(',')[0])
            cy = int(row[5].split('"cy":')[1].split('}')[0])
            keypoints.append((cx, cy))
    return keypoints

csv_path = 'path_to_your_csv_file.csv'
keypoints_data = extract_keypoints_from_csv(csv_path)

# 2. Generate Heatmaps and Vectormaps
height, width = 640, 480  # Adjust based on your image size
heatmap = generate_heatmap(height, width, keypoints_data)

# For vectormaps, you'll need to define pairs of keypoints that are connected.
# Here's an example assuming keypoints are in the order: [Rhand, Lelbow, Lshoulder, ...]
keypoints_pairs = [(keypoints_data[0], keypoints_data[1]), (keypoints_data[1], keypoints_data[2]), ...]  # Define more pairs as needed
vectormap = generate_vectormap(height, width, keypoints_pairs)
