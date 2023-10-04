from utils.image_processing import load_image, preprocess_image, apply_adaptive_thresholding, apply_otsu_thresholding, apply_color_clustering
from utils.visualization import display_images
from model import generate_vector_maps


def main():
    image_path = 'test_images/test.PNG'
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
    
    thresholded_image = apply_adaptive_thresholding(preprocessed_image)
    clustered_image = apply_color_clustering(preprocessed_image, k=3)
    vector_map_x, vector_map_y = generate_vector_maps(preprocessed_image)  # Mock vector maps for now

    display_images(preprocessed_image, thresholded_image, clustered_image, vector_map_x, vector_map_y)

if __name__ == "__main__":
    main()