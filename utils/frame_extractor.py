import cv2
import os

from image_processing import apply_adaptive_thresholding, apply_color_clustering

video_path = 'breaststroke.mp4'
video_title = os.path.splitext(os.path.basename(video_path))[0]  # Extracts 'breaststroke' from 'breaststroke.mp4'

original_folder = '../data/original_frames'
thresholded_folder = '../data/thresholded_frames'
clustered_folder = '../data/clustered_frames'

# Create the necessary folders
os.makedirs(original_folder, exist_ok=True)
os.makedirs(thresholded_folder, exist_ok=True)
os.makedirs(clustered_folder, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame_resized = cv2.resize(frame, (480, 640))

    # Save the original frame
    original_filename = os.path.join(original_folder, f'{video_title}_frame_{frame_number}.jpg')
    cv2.imwrite(original_filename, frame_resized)

    # Save the thresholded frame
    thresholded_frame = apply_adaptive_thresholding(frame_resized)
    thresholded_filename = os.path.join(thresholded_folder, f'{video_title}_frame_{frame_number}.jpg')
    cv2.imwrite(thresholded_filename, thresholded_frame)

    # Save the clustered frame
    clustered_frame = apply_color_clustering(frame_resized)
    clustered_filename = os.path.join(clustered_folder, f'{video_title}_frame_{frame_number}.jpg')
    cv2.imwrite(clustered_filename, clustered_frame)

    frame_number += 1

cap.release()
