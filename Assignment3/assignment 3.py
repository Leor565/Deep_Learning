# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:01:39 2025

@author: leor7
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

# Load the fashion_mnist dataset from TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Store the training and test sets in dictionaries
train_leor = {'images': train_images, 'labels': train_labels}
test_leor = {'images': test_images, 'labels': test_labels}

# Normalize images to range [0,1]
unsupervised_leor = {'images': train_images.astype("float32") / 255.0}
supervised_leor = {'images': test_images.astype("float32") / 255.0}

# Display dataset shapes
print(f"Training data shape: {train_leor['images'].shape}, Labels: {train_leor['labels'].shape}")
print(f"Testing data shape: {test_leor['images'].shape}, Labels: {test_leor['labels'].shape}")

# Expand dimensions for CNN input
unsupervised_leor['images'] = np.expand_dims(unsupervised_leor['images'], axis=-1)
supervised_leor['images'] = np.expand_dims(supervised_leor['images'], axis=-1)

class SampleLayer(Layer):
    def call(self, inputs):
        z_mu, z_log_sigma = inputs
        batch = tf.shape(z_mu)[0]
        dim = tf.shape(z_mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mu + tf.exp(0.5 * z_log_sigma) * epsilon  # Reparameterization trick

# Encoder architecture
def build_encoder(input_shape=(28, 28, 1)):
    input_img = Input(shape=input_shape, name='input_image')
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    feature_map_shape = x.shape[1:]  # Capture feature map size for decoder reshaping
    
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    
    z_mu_leor = Dense(2, name='z_mu_leor')(x)
    z_log_sigma_leor = Dense(2, name='z_log_sigma_leor')(x)
    
    z_leor = SampleLayer(name='z_leor')([z_mu_leor, z_log_sigma_leor])
    
    encoder = Model(input_img, [z_mu_leor, z_leor], name='encoder')
    
    return encoder, feature_map_shape, z_mu_leor, z_log_sigma_leor, z_leor  

encoder, feature_map_shape, z_mu_leor, z_log_sigma_leor, z_leor = build_encoder()
encoder.summary()

# Decoder architecture
def build_decoder(latent_dim=2, feature_map_shape=(14, 14, 64)):
    latent_inputs = Input(shape=(latent_dim,), name="z_decoder_input")
    flattened_dim = int(np.prod(feature_map_shape))
    
    x = Dense(flattened_dim, activation="relu")(latent_inputs)
    x = Reshape(feature_map_shape)(x)
    x = Conv2DTranspose(32, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
    
    decoder_outputs = Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(x)
    
    decoder_leor = Model(latent_inputs, decoder_outputs, name="decoder_leor")
    
    return decoder_leor

decoder_leor = build_decoder(feature_map_shape=feature_map_shape)
decoder_leor.summary()

# Train the VAE model
vae_leor = Model(inputs=encoder.input, outputs=decoder_leor(z_leor), name="vae_leor")
vae_leor.compile(optimizer='adam', loss='mean_squared_error')
vae_leor.fit(unsupervised_leor['images'], unsupervised_leor['images'], epochs=10, batch_size=256)
vae_leor.summary()

# Generate and plot the latent space for test dataset
z_mu_test, _ = encoder.predict(supervised_leor['images'])
plt.figure(figsize=(8, 6))
plt.scatter(z_mu_test[:, 0], z_mu_test[:, 1], c=test_leor['labels'], cmap='viridis', alpha=0.5)
plt.colorbar()
plt.xlabel("z_mu[0]")
plt.ylabel("z_mu[1]")
plt.title("Latent Space of Test Dataset")
plt.show()

# Generate 10x10 samples from the VAE model using the decoder
n = 10
figure_size = 28
norm = tfp.distributions.Normal(0, 1)
grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
figure = np.zeros((figure_size * n, figure_size * n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder_leor.predict(z_sample)
        img = x_decoded[0].reshape(figure_size, figure_size)
        figure[i * figure_size: (i + 1) * figure_size,
               j * figure_size: (j + 1) * figure_size] = img

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="gray")
plt.axis("off")
plt.title("Generated 10x10 Samples from VAE Model")
plt.show()
