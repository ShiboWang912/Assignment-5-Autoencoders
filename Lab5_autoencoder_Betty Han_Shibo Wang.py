"""
Created on Mon Nov  4 11:07:53 2024

@author: Betty Han 301202325 & Shibo Wang 301200419

"""

#1. Retrieve and load the Olivetti faces dataset 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, models
import tensorflow as tf
olivetti_faces = fetch_olivetti_faces()
olivetti_faces.data.shape

X = olivetti_faces.data  
y = olivetti_faces.target 

#2. Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set. [0 points]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
X_train.shape


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
X_train_pca.shape

#Define the Autoencoder Architecture
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def create_autoencoder(input_dim, hidden_units_1, hidden_units_2, hidden_units_3, learning_rate, l2_reg):
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder
    hidden_layer_1 = Dense(hidden_units_1, activation='relu', kernel_regularizer=l2(l2_reg))(input_layer)
    central_layer = Dense(hidden_units_2, activation='relu', kernel_regularizer=l2(l2_reg))(hidden_layer_1)
    # Decoder
    hidden_layer_3 = Dense(hidden_units_3, activation='relu', kernel_regularizer=l2(l2_reg))(central_layer)
    output_layer = Dense(input_dim, activation='sigmoid')(hidden_layer_3)

    # Autoencoder
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return autoencoder

#K-fold Cross Validation for Hyperparameter Tuning
from sklearn.model_selection import KFold
import numpy as np

# Parameters to try
hidden_units_1_values = [128, 256]
hidden_units_2_values = [64, 128]
hidden_units_3_values = [32, 64]
learning_rates = [0.001, 0.01]
l2_regs = [0.0001, 0.001]

kf = KFold(n_splits=5)
best_model = None
best_loss = float('inf')
best_params = None

for hidden_units_1 in hidden_units_1_values:
    for hidden_units_2 in hidden_units_2_values:
        for hidden_units_3 in hidden_units_3_values:
            for lr in learning_rates:
                for l2_reg in l2_regs:
                    val_losses = []

                    for train_index, val_index in kf.split(X_train_pca):
                        X_train_fold, X_val_fold = X_train_pca[train_index], X_train_pca[val_index]
                        
                        autoencoder = create_autoencoder(input_dim=X_train_pca.shape[1],
                                                         hidden_units_1=hidden_units_1,
                                                         hidden_units_2=hidden_units_2,
                                                         hidden_units_3=hidden_units_3,
                                                         learning_rate=lr,
                                                         l2_reg=l2_reg)
                        
                        history = autoencoder.fit(X_train_fold, X_train_fold,
                                                  epochs=50,
                                                  batch_size=32,
                                                  validation_data=(X_val_fold, X_val_fold),
                                                  verbose=0)
                        
                        val_loss = np.mean(history.history['val_loss'])
                        val_losses.append(val_loss)
                    
                    mean_val_loss = np.mean(val_losses)
                    
                    if mean_val_loss < best_loss:
                        best_loss = mean_val_loss
                        best_model = autoencoder
                        best_params = (hidden_units_1, hidden_units_2, hidden_units_3, lr, l2_reg)

print(f"Best Parameters: {best_params}")
#Best Parameters: (256, 128, 32, 0.01, 0.0001)

# Unpack the best parameters
hidden_units_1, hidden_units_2, hidden_units_3, lr, l2_reg = best_params

# Create and train the best model
best_autoencoder = create_autoencoder(input_dim=X_train_pca.shape[1],
                                      hidden_units_1=hidden_units_1,
                                      hidden_units_2=hidden_units_2,
                                      hidden_units_3=hidden_units_3,
                                      learning_rate=lr,
                                      l2_reg=l2_reg)

history = best_autoencoder.fit(X_train_pca, X_train_pca,
                               epochs=50,
                               batch_size=32,
                               validation_data=(X_val_pca, X_val_pca),
                               shuffle=True)


# Evaluate on the test set
test_loss = best_autoencoder.evaluate(X_test_pca, X_test_pca)
print(f'Test Loss: {test_loss}')

#Display the Original and Reconstructed Images

import matplotlib.pyplot as plt

# Function to plot original and reconstructed images
def plot_reconstruction(model, original_data, pca, n_images=10):
    reconstructed_data = model.predict(original_data)
    reconstructed_data = pca.inverse_transform(reconstructed_data)  # Inverse the PCA transformation
    original_data = pca.inverse_transform(original_data)
    
    plt.figure(figsize=(20, 4))
    
    for i in range(n_images):
        # Original images
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(original_data[i].reshape(64, 64))
        plt.gray()
        ax.axis('off')
        
        # Reconstructed images
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructed_data[i].reshape(64, 64))
        plt.gray()
        ax.axis('off')
    
    plt.show()

# Plot the original and reconstructed images from the test set
plot_reconstruction(best_autoencoder, X_test_pca, pca)

best_autoencoder.summary()