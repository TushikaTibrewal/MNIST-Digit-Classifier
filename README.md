# MNIST Digit Classifier

A simple neural network project that classifies handwritten digits from the MNIST dataset.

## Project Overview

The model takes handwritten digit images and predicts which digit they represent, from 0 to 9.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Model

The neural network contains:

- Flatten layer
- Dense layer with 128 neurons and ReLU activation
- Dense output layer with 10 neurons and Softmax activation

Architecture:

```text
28 × 28 Image
      ↓
   Flatten
      ↓
  128 Neurons
      ↓
   10 Neurons
      ↓
 Digit Prediction