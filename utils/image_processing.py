import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(image_path):
    return cv2.imread(image_path)

def preprocess_image(image):
    return image

def apply_adaptive_thresholding(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

def apply_otsu_thresholding(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def apply_color_clustering(image, k=3):
    image_reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k).fit(image_reshaped)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered.reshape(image.shape).astype(np.uint8)
    return clustered_image

def display_images(original_image, vector_map_x, vector_map_y):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(vector_map_x, cmap='viridis')
    axs[1].set_title('Vector Map X')
    axs[1].axis('off')
    
    axs[2].imshow(vector_map_y, cmap='viridis')
    axs[2].set_title('Vector Map Y')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
