import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Function to load and preprocess the dataset
def load_and_preprocess_data():
    # Load the Fashion MNIST dataset
    fmnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fmnist.load_data()

    # Normalize the images to values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


# Function to build a neural network model
def build_model():
    # Define a Sequential model with a Flatten layer and two Dense layers
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 1D vector
        tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Dense layer with 128 units and ReLU activation
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        # Output layer with 10 units for classification (softmax activation)
    ])

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Function to demonstrate how the softmax activation works
def demonstrate_softmax():
    inputs = np.array([[1.0, 3.0, 4.0, 2.0]])  # Example input array
    inputs_tensor = tf.convert_to_tensor(inputs)

    # Print the input array
    print(f'Input to softmax function: {inputs_tensor.numpy()}')

    # Apply the softmax activation function
    outputs = tf.keras.activations.softmax(inputs_tensor)

    # Print the output of the softmax function (probabilities)
    print(f'Output of softmax function: {outputs.numpy()}')

    # Calculate and print the sum of the outputs (should be 1.0)
    sum_outputs = tf.reduce_sum(outputs)
    print(f'Sum of outputs: {sum_outputs.numpy()}')

    # Find the class with the highest probability
    prediction = np.argmax(outputs)
    print(f'Class with highest probability: {prediction}')


# Function to visualize a specific image from the training data
def visualize_image(train_images, train_labels, index=42):
    # Set number of characters per row when printing
    np.set_printoptions(linewidth=320)

    # Print the label and image
    print(f'LABEL: {train_labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n {train_images[index]}')

    # Visualize the image using Matplotlib
    plt.imshow(train_images[index], cmap='gray')
    plt.title(f'Label: {train_labels[index]}')
    plt.show()


# Main function to run the entire workflow
def main():
    # Load and preprocess the dataset
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    # Build and compile the model
    model = build_model()

    # Demonstrate softmax on a small example
    demonstrate_softmax()

    # Train the model for 5 epochs
    model.fit(train_images, train_labels, epochs=5)

    # Visualize a sample image from the training set
    visualize_image(train_images, train_labels)


# Entry point of the script
if __name__ == "__main__":
    main()

