import cv2
import tensorflow as tf
from data_generator import DataGenerator
import keras.layers
import numpy as np
import matplotlib.pyplot as plt


def create_model(input_shape, num_keypoints):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)

    # Heatmap output with Transposed Convolution and Refinement
    heatmap_output = keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    heatmap_output = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(heatmap_output)
    heatmap_output = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(heatmap_output)
    heatmap_output = keras.layers.Conv2DTranspose(num_keypoints, (4, 4), strides=(2, 2), padding='same', activation='sigmoid', name='heatmap_output')(heatmap_output)

    # Vectormap output with Transposed Convolution and Refinement
    vectormap_output = keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    vectormap_output = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(vectormap_output)
    vectormap_output = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(vectormap_output)
    vectormap_output = keras.layers.Conv2DTranspose(2*num_keypoints, (4, 4), strides=(2, 2), padding='same', activation='tanh', name='vectormap_output')(vectormap_output)

    model = keras.Model(inputs=inputs, outputs=[heatmap_output, vectormap_output])
    return model

def visualize_results(original_image, predicted_heatmap, predicted_vectormap):
    plt.figure(figsize=(15, 5))

    # Extract keypoints from the heatmap
    keypoints = []
    for i in range(predicted_heatmap.shape[-1]):
        y, x = np.unravel_index(np.argmax(predicted_heatmap[:, :, i]), predicted_heatmap[:, :, i].shape)
        keypoints.append((x, y))

    # Display original image with keypoints
    plt.subplot(1, 3, 1)
    img_with_keypoints = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB).copy()
    for x, y in keypoints:
        cv2.circle(img_with_keypoints, (x, y), 3, (0, 255, 0), -1)  # Green color for keypoints
    plt.imshow(img_with_keypoints)
    plt.title('Original Image with Predicted Keypoints')

    # Display combined predicted heatmap
    combined_heatmap = np.sum(predicted_heatmap, axis=-1)
    plt.subplot(1, 3, 2)
    plt.imshow(combined_heatmap, cmap='hot')
    plt.title('Combined Predicted Heatmap')

    # Display combined predicted vectormap
    combined_vectormap = np.sum(predicted_vectormap, axis=-1)
    plt.subplot(1, 3, 3)
    plt.imshow(combined_vectormap, cmap='hot')
    plt.title('Combined Predicted Vectormap')

    plt.show()


if __name__ == "__main__":
    # Parameters
    image_dir = 'data/original_frames'
    csv_path = 'data/mappings/breaststroke.csv'
    input_shape = (240, 320, 3)  # Adjusted to half of the previous size

    # Assuming there are 14 keypoints (based on the connections you provided)
    num_keypoints = 14

    # Create model
    model = create_model(input_shape, num_keypoints)
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], metrics=['accuracy'])

    # Initialize data generator
    data_gen = DataGenerator(image_dir, csv_path)

    # Train the model
    model.fit(data_gen, epochs=1)

    test_image_path = 'test_images/test.PNG'
    test_image = cv2.imread(test_image_path)
    original_shape = test_image.shape
    test_image = cv2.resize(test_image, (320, 240))  # Resize to the input shape expected by the model
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    predicted_heatmap, predicted_vectormap = model.predict(test_image)
    visualize_results(test_image[0], predicted_heatmap[0], predicted_vectormap[0])

