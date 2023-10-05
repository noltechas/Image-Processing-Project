import tensorflow as tf
from data_generator import DataGenerator
import keras.layers


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

    # Adjust the number of filters in the last Conv2D layer to match the number of keypoints
    heatmap_output = keras.layers.Conv2D(num_keypoints, (1, 1), activation='sigmoid', name='heatmap_output')(x)
    heatmap_output = keras.layers.UpSampling2D(size=(16, 16))(heatmap_output)  # Upsample to match the input shape

    vectormap_output = keras.layers.Conv2D(2*num_keypoints, (1, 1), activation='tanh', name='vectormap_output')(x)
    vectormap_output = keras.layers.UpSampling2D(size=(16, 16))(vectormap_output)  # Upsample to match the input shape

    model = keras.Model(inputs=inputs, outputs=[heatmap_output, vectormap_output])
    return model


if __name__ == "__main__":
    # Parameters
    image_dir = 'data/original_frames'
    csv_path = 'data/mappings/breaststroke.csv'
    input_shape = (None, None, 3)  # Height, Width, Channels

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
    model.fit(data_gen, epochs=10)
