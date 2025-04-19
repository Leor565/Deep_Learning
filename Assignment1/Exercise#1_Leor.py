"""
Created on Mon Jan 27 12:06:28 2025

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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Organize into dictionaries
train_leor = {'images': train_images, 'labels': train_labels}
test_leor = {'images': test_images, 'labels': test_labels}

print(f"Train data keys: {list(train_leor.keys())}")
print(f"Test data keys: {list(test_leor.keys())}")
print(f"Train images shape: {train_leor['images'].shape}")
print(f"Train labels shape: {train_leor['labels'].shape}")
print(f"Test images shape: {test_leor['images'].shape}")
print(f"Test labels shape: {test_leor['labels'].shape}")

print(f"Number of training examples: {len(train_leor['images'])}")
print(f"Number of test examples: {len(test_leor['images'])}")

print(f"Images resolution: {train_leor['images'].shape[1]} x {train_leor['images'].shape[2]}")

max_pixel_value = np.amax(train_leor['images'])
print(f"Largest pixel value in the dataset: {max_pixel_value}")

# Normalize the images to the range [0, 1]
train_leor['images'] = train_leor['images'] / 255.0
test_leor['images'] = test_leor['images'] / 255.0

# One-hot encode the labels (store separately)
train_labels_one_hot = to_categorical(train_leor['labels'])
test_labels_one_hot = to_categorical(test_leor['labels'])

# Print the shape of the one-hot encoded labels
print(f"train labels one-hot shape: {train_labels_one_hot.shape}")
print(f"test labels one-hot shape: {test_labels_one_hot.shape}")

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
plot_first_12_images(train_leor['images'], train_labels_decoded)

# Split the data for training and validation
validation_split = 0.2
input_train, input_val, target_train, target_val = train_test_split(
    train_leor['images'], train_leor['labels'], test_size=validation_split, random_state=36
)

# Store training features and labels in DataFrames
x_train_leor = pd.DataFrame(input_train.reshape(input_train.shape[0], -1))  # Flatten images to 1D
y_train_leor = pd.DataFrame(target_train)

# Store validation features and labels in DataFrames
x_val_leor = pd.DataFrame(input_val.reshape(input_val.shape[0], -1))  # Flatten images to 1D
y_val_leor = pd.DataFrame(target_val)

# Ensure one-hot encoding for validation labels as well
y_val_leor_one_hot = to_categorical(y_val_leor)

# Check the DataFrames
print(f"x_train_leor shape: {x_train_leor.shape}")
print(f"y_train_leor shape: {y_train_leor.shape}")
print(f"x_val_leor shape: {x_val_leor.shape}")
print(f"y_val_leor shape: {y_val_leor.shape}")

# Model parameters
input_shape = (28, 28, 1)
num_classes = 10

# Create the CNN model
model = Sequential(name="cnn_model_leor")
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))  # First layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Second layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))  # Third layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Fourth layer
model.add(Flatten())  # Flattening the image
model.add(Dense(100, activation='sigmoid'))  # Fifth layer with sigmoid activation
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),  # Default Adam optimizer
    metrics=['accuracy']
)

# Model summary
print(model.summary())

# Train the model
cnn_history_leor = model.fit(
    train_leor['images'],  # Training data reshaped to (None, 28, 28, 1)
    train_labels_one_hot,  # One-hot encoded training labels
    batch_size=256,        # Batch size
    epochs=8,              # Number of epochs
    validation_data=(input_val, y_val_leor_one_hot)  # One-hot encoded validation labels
)

# Evaluate the model with the test dataset
test_loss, test_accuracy = model.evaluate(test_leor['images'], test_labels_one_hot, verbose=0)

# Print the test accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot Training vs Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(cnn_history_leor.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(cnn_history_leor.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Generate predictions on the test dataset
cnn_predictions_leor = model.predict(test_leor['images'])

# Function to plot the probability distribution for predictions
def plot_prediction_distribution(true_label, predicted_probabilities):
    """
    Plots the probability distribution for predictions.

    Parameters:
        true_label (int): The true label index.
        predicted_probabilities (array): Array of probabilities for each class.
    """
    num_classes = len(predicted_probabilities)
    bar_colors = ['blue'] * num_classes
    bar_colors[true_label] = 'green'  # Color the true label in green

    plt.bar(range(num_classes), predicted_probabilities, color=bar_colors)
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Class Index')
    plt.ylabel('Probability')
    plt.xticks(range(num_classes))
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

# Example: Call the function for a test sample
sample_index = 0  # Replace with any test sample index
true_label = test_labels_one_hot[sample_index].argmax()  # True label of the sample
predicted_probabilities = cnn_predictions_leor[sample_index]  # Predicted probabilities for the sample

plot_prediction_distribution(true_label, predicted_probabilities)




class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Function to display the image, its true label, and the prediction probability distribution
def display_images_and_predictions(test_images, test_labels, predictions, start_index):
    num_images = 4
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

# Function to plot prediction probability distribution
def plot_prediction_distribution(true_label, predicted_probabilities):
    # Plot the distribution of predicted probabilities for each class
    plt.bar(range(10), predicted_probabilities, color='gray')
    plt.xticks(range(10), class_names, rotation=90)
    plt.xlabel('Clothing Class')
    plt.ylabel('Probability')
    plt.title(f"Predicted Label: {class_names[true_label]} (True Label)")

# Start index based on the last two digits of your student number (assume 23 -> start from index 24)
start_index = 36 # Starting from image 24 (index 23)

# Assuming cnn_predictions_leor contains the prediction probabilities
# Display the images and predictions
display_images_and_predictions(test_leor['images'], test_leor['labels'], cnn_predictions_leor, start_index)


import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a heatmap from the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Add labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Display the plot
    plt.show()

# The test labels are in test_leor['labels']
# The predictions are a probability distribution, so we take the argmax to get the predicted class label
y_true = test_leor['labels']
y_pred = np.argmax(cnn_predictions_leor, axis=1)  # Predicted class label (index of highest probability)

# Plot the confusion matrix
plot_confusion_matrix(y_true, y_pred, class_names)