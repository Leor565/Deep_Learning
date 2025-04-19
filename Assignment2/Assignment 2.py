# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:28:48 2025
@author: leor7
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Step 1: Load the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Step 2: Normalize the pixel values (0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 3: Store datasets in dictionaries
unsupervised_leor = {"images": x_train}  # First 60,000 samples
supervised_leor = {"images": x_test, "labels": y_test}  # Next 10,000 samples

# Verify dataset shapes
print("Unsupervised dataset shape:", unsupervised_leor["images"].shape)
print("Supervised dataset shapes:", supervised_leor["images"].shape, supervised_leor["labels"].shape)

# Step 4: One-hot encode supervised labels
supervised_leor["labels"] = to_categorical(y_test)

# Verify one-hot encoding
print("Shape of one-hot encoded labels:", supervised_leor["labels"].shape)
print(supervised_leor["labels"][:5])

# Step 5: Split unsupervised dataset (60,000) into:
# - Training: 57,000
# - Validation: 3,000
student_id = 36  
x_train_unsupervised = unsupervised_leor["images"]

x_train_unsup, x_val_unsup = train_test_split(
    x_train_unsupervised, test_size=3000, random_state=student_id
)

# Convert to DataFrames
unsupervised_train_leor = pd.DataFrame(x_train_unsup.reshape(x_train_unsup.shape[0], -1))
unsupervised_val_leor = pd.DataFrame(x_val_unsup.reshape(x_val_unsup.shape[0], -1))

# Verify shapes
print("Training set shape:", unsupervised_train_leor.shape)  # Expected: (57000, 784)
print("Validation set shape:", unsupervised_val_leor.shape)  # Expected: (3000, 784)

# Step 6: Keep only 3,000 samples from supervised dataset (discard 7,000)
x_supervised = supervised_leor["images"]
y_supervised = supervised_leor["labels"]

x_supervised, _, y_supervised, _ = train_test_split(
    x_supervised, y_supervised, test_size=0.7, random_state=student_id
)

# Step 7: Split remaining supervised dataset (3,000) into:
# - Training: 1,800
# - Validation: 600
# - Testing: 600
x_train_sup, x_temp, y_train_sup, y_temp = train_test_split(x_supervised, y_supervised, test_size=1200, random_state=student_id)

x_val_sup, x_test_sup, y_val_sup, y_test_sup = train_test_split(x_temp, y_temp, test_size=600, random_state=student_id)

# Step 8: Store datasets as DataFrames
x_train_leor = pd.DataFrame(x_train_sup.reshape(x_train_sup.shape[0], -1))
x_val_leor = pd.DataFrame(x_val_sup.reshape(x_val_sup.shape[0], -1))
x_test_leor = pd.DataFrame(x_test_sup.reshape(x_test_sup.shape[0], -1))

y_train_leor = pd.DataFrame(y_train_sup)
y_val_leor = pd.DataFrame(y_val_sup)
y_test_leor = pd.DataFrame(y_test_sup)

# Final verification
print("x_train_leor shape:", x_train_leor.shape)  # Expected: (1800, 784)
print("x_val_leor shape:", x_val_leor.shape)      # Expected: (600, 784)
print("x_test_leor shape:", x_test_leor.shape)    # Expected: (600, 784)

print("y_train_leor shape:", y_train_leor.shape)  # Expected: (1800, 10)
print("y_val_leor shape:", y_val_leor.shape)      # Expected: (600, 10)
print("y_test_leor shape:", y_test_leor.shape)    # Expected: (600, 10)


#Part D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

x_train_leor = x_train_leor.values.reshape(-1, 28, 28, 1)
x_val_leor = x_val_leor.values.reshape(-1, 28, 28, 1)
x_test_leor = x_test_leor.values.reshape(-1, 28, 28, 1)

# Define the CNN model
cnn_v1_model_leor = Sequential([
    # 1st Convolutional Layer: 16 filters, 3x3 kernel, stride=2, 'same' padding
    Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
    
    # Max Pooling Layer (2x2)
    MaxPooling2D(pool_size=(2, 2)),

    # 2nd Convolutional Layer: 64 filters, 3x3 kernel, ReLU activation
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

    # Max Pooling Layer (2x2)
    MaxPooling2D(pool_size=(2, 2)),

    # 3rd Convolutional Layer: 8 filters, 3x3 kernel, stride=2, 'same' padding
    Conv2D(filters=8, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),

    # Flatten Layer (converts 2D feature maps to 1D)
    Flatten(),

    # 4th Fully Connected Layer: 100 neurons, ReLU activation
    Dense(100, activation='relu'),

    # Output Layer (10 classes for Fashion MNIST)
    Dense(10, activation='softmax')  
])

# Compile the model
cnn_v1_model_leor.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("SUMMARY")
# Display the model summary
cnn_v1_model_leor.summary()

# Set the batch size and epochs
batch_size = 256
epochs = 10

# Train the model using the fit() method
cnn_v1_history_leor = cnn_v1_model_leor.fit(
    x_train_leor, 
    y_train_leor, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(x_val_leor, y_val_leor)
)
    
#Part E
import matplotlib.pyplot as plt



# Extract accuracy from the training history
train_acc = cnn_v1_history_leor.history['accuracy']
val_acc = cnn_v1_history_leor.history['val_accuracy']

# Plot the training and validation accuracy
plt.plot(train_acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='red')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')

# Add legend
plt.legend()
# Show the plot
plt.show()

# Evaluate the model on the test dataset
test_loss, test_acc = cnn_v1_model_leor.evaluate(x_test_leor, y_test_leor)

# Print the test accuracy
print("Test Accuracy:", test_acc)
val_acc = cnn_v1_history_leor.history['val_accuracy'][-1]  # Last validation accuracy value

# Print the validation accuracy
print("Validation Accuracy:", val_acc)


cnn_predictions_leor = cnn_v1_model_leor.predict(x_test_leor)
print("Predictions shape:", cnn_predictions_leor.shape)
print("First 5 predictions", cnn_predictions_leor[:5])


from sklearn.metrics import confusion_matrix
import seaborn as sns

cnn_predictions_leor = cnn_v1_model_leor.predict(x_test_leor)
cnn_predictions_leor = np.argmax(cnn_predictions_leor, axis=1)
cm = confusion_matrix(np.argmax(y_test_leor, axis= 1), cnn_predictions_leor)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual Label")
plt.show()

#Part F 

tf.random.set_seed(student_id)
noise_factor = 0.2
noise_train = tf.random.normal(shape=x_train_unsup.shape, mean=0.0, stddev=noise_factor)
noise_val = tf.random.normal(shape=x_val_unsup.shape, mean=0.0, stddev=noise_factor)

# Step 2: Add the noise to the unsupervised datasets
x_train_noisy_leor = x_train_unsup + noise_train
x_val_noisy_leor = x_val_unsup + noise_val

# Step 3: Clip the values to keep them within valid pixel range [0, 255] or [0, 1]
x_train_noisy_leor = tf.clip_by_value(x_train_noisy_leor, 0.0, 1.0)
x_val_noisy_leor = tf.clip_by_value(x_val_noisy_leor, 0.0, 1.0)

# Verify the shapes and data range
print("x_train_noisy_leor shape:", x_train_noisy_leor.shape)
print("x_val_noisy_leor shape:", x_val_noisy_leor.shape)
print("Value range in noisy train data:", tf.reduce_min(x_train_noisy_leor).numpy(), "-", tf.reduce_max(x_train_noisy_leor).numpy())




# Clip values to be in the range [0, 1]
x_train_noisy_leor = tf.clip_by_value(x_train_noisy_leor, 0.0, 1.0)
x_val_noisy_leor = tf.clip_by_value(x_val_noisy_leor, 0.0, 1.0)

# Verify the value range after clipping
print("Clipped Value Range in Noisy Train Data:", tf.reduce_min(x_train_noisy_leor).numpy(), "-", tf.reduce_max(x_train_noisy_leor).numpy())
print("Clipped Value Range in Noisy Validation Data:", tf.reduce_min(x_val_noisy_leor).numpy(), "-", tf.reduce_max(x_val_noisy_leor).numpy())


# Display the first 10 noisy validation images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)  # Create a 2-row, 5-column subplot
    plt.imshow(x_val_noisy_leor[i], cmap="gray") 
    plt.axis("off") 

plt.suptitle("First 10 Noisy Validation Images", fontsize=14)
plt.show()


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Step 1: Define the autoencoder architecture
inputs_leor = Input(shape=(28, 28, 1))  # Fashion MNIST image size
e_leor = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(inputs_leor)
e_leor = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(e_leor)

d_leor = Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2)(e_leor)
d_leor = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2)(d_leor)
d_leor = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d_leor)

# Build the autoencoder model
autoencoder_leor = Model(inputs=inputs_leor, outputs=d_leor)

# Step 2: Compile the model
autoencoder_leor.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Display the model summary
autoencoder_leor.summary()

autoencoder_leor.fit(x_train_noisy_leor, x_train_unsup, epochs=10, batch_size=256, shuffle=True,
                     validation_data=(x_val_noisy_leor, x_val_unsup))


# Step 1: Generate predictions on the unsupervised_val_leor dataset
autoencoder_predictions_leor = autoencoder_leor.predict(x_val_noisy_leor)

# Step 2: Display the first 10 predicted images
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i+1)  # 2 rows, 5 columns
    image = np.mean(autoencoder_predictions_leor[i], axis=-1)  # Use mean() to remove the 3rd axis if necessary
    plt.imshow(image, cmap='gray')  
    plt.axis('off')  

plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# Step 1: Define the architecture for cnn_v2_firstname
encoder_model_leor= Model(inputs=autoencoder_leor.input, outputs=e_leor)

cnn_v2_leor = Sequential([
    encoder_model_leor,  # Using the encoder from Autoencoder
    Flatten(),  # Flatten the feature maps
    Dense(100, activation='relu'),  # Fully connected layer with 100 neurons
    Dense(10, activation='softmax')  # Output layer with 10 classes 
])

cnn_v2_leor.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_v2_leor.summary()
cnn_v2_history_leor = cnn_v2_leor.fit(
    x_train_leor.reshape(-1, 28, 28, 1), y_train_leor,
    validation_data=(x_val_leor.reshape(-1, 28, 28, 1), y_val_leor),
    epochs=10, batch_size=256
)




plt.figure(figsize=(8, 5))
plt.plot(cnn_v2_history_leor.history['accuracy'], label='Train Accuracy', linestyle='-', color='blue')
plt.plot(cnn_v2_history_leor.history['val_accuracy'], label='Validation Accuracy', linestyle='--', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy - Transfer Learning CNN')
plt.show()



test_loss, test_accuracy = cnn_v2_leor.evaluate(x_test_leor.reshape(-1, 28, 28, 1), y_test_leor)
print(f"Test Accuracy of Transfer Learning CNN: {test_accuracy:.4f}")


cnn_predictions_leor = np.argmax(cnn_v2_leor.predict(x_test_leor.reshape(-1, 28, 28, 1)), axis=1)
true_labels_leor = np.argmax(y_test_leor, axis=1)



conf_matrix_leor = confusion_matrix(true_labels_leor, cnn_predictions_leor)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_leor, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Transfer Learning CNN')
plt.show()



plt.figure(figsize=(8, 5))
plt.plot(cnn_v1_history_leor.history['val_accuracy'], label='Baseline CNN', linestyle='-', color='blue')
plt.plot(cnn_v2_history_leor.history['val_accuracy'], label='Transfer Learning CNN', linestyle='--', color='red')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy: Baseline CNN vs Transfer Learning CNN')
plt.show()



# Evaluate test accuracy for both models
test_loss_baseline, test_accuracy_baseline = cnn_v1_model_leor.evaluate(
    x_test_leor.reshape(-1, 28, 28, 1), y_test_leor, verbose=0)

test_loss_transfer, test_accuracy_transfer = cnn_v2_leor.evaluate(
    x_test_leor.reshape(-1, 28, 28, 1), y_test_leor, verbose=0)

print(f"Test Accuracy - Baseline CNN: {test_accuracy_baseline:.4f}")
print(f"Test Accuracy - Transfer Learning CNN: {test_accuracy_transfer:.4f}")


if test_accuracy_transfer > test_accuracy_baseline:
    print("The Transfer Learning CNN performed better in terms of accuracy.")
elif test_accuracy_transfer < test_accuracy_baseline:
    print("The Baseline CNN performed better in terms of accuracy.")
else:
    print("Both models have similar test accuracy.")
