import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)

# Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig("training_curve.png")
plt.show()

# Make predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Find misclassified images
wrong_predictions = np.where(predicted_labels != y_test)[0]

print("Number of misclassified images:", len(wrong_predictions))

# Display some misclassified digits
plt.figure(figsize=(10, 5))

for i in range(10):
    index = wrong_predictions[i]

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[index], cmap="gray")
    plt.title(
        f"Actual: {y_test[index]}\nPredicted: {predicted_labels[index]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()