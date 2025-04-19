# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:51:23 2025

@author: leor7
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Split the data for training and validation
validation_split = 0.2
input_train, input_val, target_train, target_val = train_test_split(
    train_images, train_labels_one_hot, test_size=validation_split, random_state=36
)

# Model parameters
num_classes = 10
input_shape = (28, 28, 1)  # Each image is 28x28, LSTM takes sequences

# Define the LSTM model
model = Sequential([
    LSTM(128, input_shape=(28, 28)),  # LSTM treats each row as a timestep
    Dense(num_classes, activation='softmax')  # Output layer with softmax
])

# Compile the model
# Compile the model with categorical_crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Function to plot a single image with its label
def plot_image_with_label(image, label):
    plt.imshow(image, cmap='gray')  # Display the image in grayscale
    plt.title(f"Label: {label}")
    plt.axis('off')  # Remove axes for a cleaner plot

# Function to plot the first 12 images in a 4x3 grid
def plot_first_12_images(images, labels):
    plt.figure(figsize=(8, 8))  # Set figure size to 8x8
    for i in range(12):
        plt.subplot(4, 3, i + 1)  # Create a 4x3 grid of subplots
        plot_image_with_label(images[i], labels[i])  # Plot each image with its label
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

# Decode the one-hot labels back to integers
train_labels_decoded = [label.argmax() for label in train_labels_one_hot]

# Plot the first 12 images with their decoded labels
plot_first_12_images(train_images, train_labels_decoded)

# Train the model
history = model.fit(
    input_train, target_train,  # Training data
    batch_size=256,  # Batch size
    epochs=8,  # Number of epochs
    validation_data=(input_val, target_val)  # Validation data
)

# Plot loss and accuracy curves
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call function to plot loss and accuracy
plot_history(history)

# Evaluate the model with the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot, verbose=0)

# Print the test accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions on the test dataset
predictions = model.predict(test_images)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Convert predictions to class labels (argmax)
predicted_labels = np.argmax(predictions, axis=1)

# Class names for Fashion MNIST dataset
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Plot confusion matrix
plot_confusion_matrix(test_labels, predicted_labels, class_names)

# Function to plot the prediction probability distribution for a given image
def plot_prediction_distribution(predicted_label, predicted_probabilities):
    plt.bar(range(10), predicted_probabilities, color='skyblue')
    plt.xticks(range(10), class_names, rotation=45)
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title(f"Predicted: {class_names[predicted_label]}")
    plt.ylim(0, 1)

# Function to display the image, its true label, and the prediction probability distribution
def display_images_and_predictions(test_images, test_labels, predictions, start_index):
    num_images = 4  # Display 4 images
    for i in range(num_images):
        # Get the image and the true label
        image = test_images[start_index + i]
        true_label = test_labels[start_index + i]
        predicted_probabilities = predictions[start_index + i]
        
        # Get predicted class label (index of the max probability)
        predicted_label = np.argmax(predicted_probabilities)
        
        # Plot the image with its true label
        plt.figure(figsize=(6, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')  # Display the image in grayscale
        plt.title(f"True Label: {class_names[true_label]}")
        plt.axis('off')  # Remove axes for a cleaner plot
        
        # Plot the prediction distribution
        plt.subplot(1, 2, 2)
        plot_prediction_distribution(predicted_label, predicted_probabilities)
        
        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

# Use the function to display images, true labels, and predictions for a specific starting index
display_images_and_predictions(test_images, test_labels, predictions, start_index=0)
