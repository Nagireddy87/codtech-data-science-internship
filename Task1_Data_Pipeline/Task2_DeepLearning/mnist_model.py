# Task 2 - Deep Learning Project
# Handwritten Digit Classification using TensorFlow

import tensorflow as tf
import matplotlib.pyplot as plt

print("Loading MNIST dataset...")

# 1️⃣ Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Dataset loaded successfully!")

# 2️⃣ Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3️⃣ Build neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4️⃣ Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

# 5️⃣ Train model
history = model.fit(x_train, y_train, epochs=5)

# 6️⃣ Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nModel Accuracy:", test_acc)

# 7️⃣ Plot training accuracy
plt.plot(history.history['accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy_plot.png")
plt.show()

# 8️⃣ Predict sample image
prediction = model.predict(x_test)

print("\nPrediction for first image:", prediction[0].argmax())
print("Actual label:", y_test[0])