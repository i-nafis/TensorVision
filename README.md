# Tensorvision

Tensorvision is a deep learning project that focuses on building and training a neural network for image classification using the Fashion MNIST dataset. This project demonstrates how to preprocess data, build a neural network using TensorFlow/Keras, and train the model to classify images of clothing.

## Features

- **Preprocessing**: Normalizes image pixel values to a range of 0-1.
- **Model**:
  - A `Flatten` layer to convert 28x28 images into a 1D vector.
  - A hidden `Dense` layer with 128 neurons using the ReLU activation function.
  - An output `Dense` layer with 10 neurons and softmax activation for classification.
- **Training**: Uses the Adam optimizer and sparse categorical crossentropy loss to train the model.
- **Softmax Example**: Demonstrates the use of the softmax function to convert logits into probabilities.
- **Visualization**: Displays a sample image from the dataset for reference.

## Dataset

The **Fashion MNIST** dataset contains 70,000 grayscale images in 10 different clothing categories. Each image is 28x28 pixels.

### Clothing Categories:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

## Model Architecture

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Flatten:** Converts 28x28 pixel images to a 784-element vector.

**Dense (Hidden Layer):** 128 neurons with ReLU activation.

**Dense (Output Layer):** 10 neurons with softmax activation for classification.
