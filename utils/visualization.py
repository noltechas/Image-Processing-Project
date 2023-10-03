import matplotlib.pyplot as plt

def display_images(original_image, thresholded_image, clustered_image, vector_map_x, vector_map_y):
    fig = plt.figure(figsize=(20, 5))

    # Display Original Image
    ax1 = fig.add_subplot(1, 5, 1)
    ax1.set_title('Original Image')
    ax1.imshow(original_image)

    # Display Thresholded Image
    ax2 = fig.add_subplot(1, 5, 2)
    ax2.set_title('Thresholded Image')
    ax2.imshow(thresholded_image, cmap='gray')

    # Display Clustered Image
    ax3 = fig.add_subplot(1, 5, 3)
    ax3.set_title('Clustered Image')
    ax3.imshow(clustered_image)

    # Display Vector Map X
    ax4 = fig.add_subplot(1, 5, 4)
    ax4.set_title('Vector Map X')
    ax4.imshow(vector_map_x, cmap='gray')

    # Display Vector Map Y
    ax5 = fig.add_subplot(1, 5, 5)
    ax5.set_title('Vector Map Y')
    ax5.imshow(vector_map_y, cmap='gray')

    plt.show()
