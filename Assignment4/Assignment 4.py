# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:09:49 2025

@author: leor7
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Flatten
from tensorflow.keras import Sequential

# a. Get the data
# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Store data in the corresponding dictionaries
ds1_leor = {'images': train_images[:60000], 'labels': train_labels[:60000]}
ds2_leor = {'images': test_images[:10000], 'labels': test_labels[:10000]}

# b. Dataset Preprocessing
# Normalize pixel values to the range -1 to 1
ds1_leor['images'] = (ds1_leor['images'].astype(np.float32) - 127.5) / 127.5
ds2_leor['images'] = (ds2_leor['images'].astype(np.float32) - 127.5) / 127.5

# Print the shapes of the images
print(ds1_leor['images'].shape)
print(ds2_leor['images'].shape)

# Concatenate the images of class 1 (pants) from both datasets
pants_images_ds1 = ds1_leor['images'][ds1_leor['labels'] == 1]
pants_images_ds2 = ds2_leor['images'][ds2_leor['labels'] == 1]

dataset_leor = np.concatenate([pants_images_ds1, pants_images_ds2], axis=0)

# Print the shape of the concatenated dataset
print(dataset_leor.shape)

# Display the first 12 images from the dataset
plt.figure(figsize=(8, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.imshow(dataset_leor[i], cmap='gray')
    plt.axis('off')
plt.show()

# Create the training dataset using TensorFlow Dataset
train_dataset_leor = tf.data.Dataset.from_tensor_slices(dataset_leor)
train_dataset_leor = train_dataset_leor.shuffle(7000).batch(256)

# c. Build the Generator Model
generator_model_leor = Sequential([
    # Input: 100-Dimensional Vector
    Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
    BatchNormalization(),
    LeakyReLU(),
    
    # Reshape to (7, 7, 256)
    Reshape((7, 7, 256)),
    
    # Transposed Convolution Layer
    Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    
    # Transposed Convolution Layer
    Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(),
    
    # Final Transposed Convolution Layer to output a single-channel image
    Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh', use_bias=False)
])

# Display the model summary
generator_model_leor.summary()

# d. Sample untrained generator
# Generate a random input vector
noise = tf.random.normal([1, 100])

# Generate an image from the generator model
generated_image = generator_model_leor(noise, training=False)

# Plot the generated image
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

# e. Build the Discriminator Model
discriminator_model_leor = Sequential([
    # Convolution Layer
    Conv2D(64, 5, strides=2, padding='same', input_shape=[28, 28, 1]),
    LeakyReLU(),
    Dropout(0.3),
    
    # Convolution Layer
    Conv2D(128, 5, strides=2, padding='same'),
    LeakyReLU(),
    Dropout(0.3),
    
    # Flatten the result
    Flatten(),
    
    # Fully Connected Layer
    Dense(1)
])

# Display the model summary
discriminator_model_leor.summary()

# f. Implement Training
cross_entropy_leor = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
generator_optimizer_leor = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer_leor = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Training step
def train_step(images):
    noise = tf.random.normal([256, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model_leor(noise, training=True)
        
        real_output = discriminator_model_leor(images, training=True)
        fake_output = discriminator_model_leor(generated_images, training=True)
        
        gen_loss = cross_entropy_leor(tf.ones_like(fake_output), fake_output)
        real_loss = cross_entropy_leor(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy_leor(tf.zeros_like(fake_output), fake_output)
        
        disc_loss = real_loss + fake_loss
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model_leor.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model_leor.trainable_variables)
    
    generator_optimizer_leor.apply_gradients(zip(gradients_of_generator, generator_model_leor.trainable_variables))
    discriminator_optimizer_leor.apply_gradients(zip(gradients_of_discriminator, discriminator_model_leor.trainable_variables))

# g. Train the models
import time

epochs = 10
for epoch in range(epochs):
    start_time = time.time()
    
    for batch in train_dataset_leor:
        train_step(batch)
    
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1} took {elapsed_time:.2f} seconds")

# h. Visualize the Trained Generator
# Generate 16 random vectors and create images
random_vectors = tf.random.normal([16, 100])
generated_images = generator_model_leor(random_vectors, training=False)

# Rescale images to the range [0, 255]
generated_images = (generated_images * 127.5) + 127.5

# Plot the generated images
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()
