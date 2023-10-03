import cv2
import os

from image_processing import apply_adaptive_thresholding, apply_color_clustering

def extract_frames_from_video(video_path, original_folder, thresholded_folder, clustered_folder):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}")
        return

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the original frame
        original_filename = os.path.join(original_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(original_filename, frame)
        
        # Save the thresholded frame
        thresholded_frame = apply_adaptive_thresholding(frame)
        thresholded_filename = os.path.join(thresholded_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(thresholded_filename, thresholded_frame)
        
        # Save the clustered frame
        clustered_frame = apply_color_clustering(frame)
        clustered_filename = os.path.join(clustered_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(clustered_filename, clustered_frame)
        
        frame_count += 1
    
    cap.release()

if __name__ == "__main__":
    video_path = '../test_images/breaststroke.mp4'  # Hardcoded path to the video
    original_folder = '../data/original_frames'  # Folder for original frames
    thresholded_folder = '../data/thresholded_frames'  # Folder for thresholded frames
    clustered_folder = '../data/clustered_frames'  # Folder for clustered frames
    extract_frames_from_video(video_path, original_folder, thresholded_folder, clustered_folder)
