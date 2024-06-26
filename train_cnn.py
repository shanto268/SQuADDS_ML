"""
1 D CNN
"""

import datetime
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sys

fname_py = sys.argv[0].split(".")[0]

# Create unique directories for saving models and figures
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
models_dir = f'models/{fname}_{timestamp}'
figures_dir = f'figures/{fname}_{timestamp}'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Load data
training_data = pd.read_csv(f"data/train_dataset_3.parquet")

# Extract input and output features
X = training_data[['qubit_frequency_GHz', 'anharmonicity_MHz', 'cavity_frequency_GHz', 'kappa_kHz', 'g_MHz']].values
y = training_data[['cross_length', 'claw_length', 'EJ', 'coupling_length', 'total_length', 'ground_spacing']].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example of creating polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Save the polynomial feature transformer
joblib.dump(poly, f'{models_dir}/poly_transformer.pkl')

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_poly = scaler_X.fit_transform(X_train_poly)
X_test_poly = scaler_X.transform(X_test_poly)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Save the scalers
joblib.dump(scaler_X, f'{models_dir}/scaler_X.pkl')
joblib.dump(scaler_y, f'{models_dir}/scaler_y.pkl')

# Reshape the data for 1D CNN input
X_train_poly = X_train_poly.reshape((X_train_poly.shape[0], X_train_poly.shape[1], 1))
X_test_poly = X_test_poly.reshape((X_test_poly.shape[0], X_test_poly.shape[1], 1))

# Define the 1D CNN model creation function
def build_cnn_model(input_shape, output_shape, learning_rate=0.001):
    inputs = Input(shape=(input_shape[0], input_shape[1]))
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(output_shape)(x)
    
    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

# Custom function for cross-validation
def cross_val_score_keras(model_fn, X, y, cv=5, epochs=100, batch_size=32):
    kf = KFold(n_splits=cv)
    val_scores = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model = model_fn()
        model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)
        val_mse = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        val_scores.append(val_mse)

    return np.mean(val_scores)

# Perform cross-validation
input_shape = X_train_poly.shape[1:]
output_shape = y_train.shape[1]
mean_cv_mse = cross_val_score_keras(lambda: build_cnn_model(input_shape, output_shape), X_train_poly, y_train, cv=2, epochs=100, batch_size=32)
print(f"Mean Cross-Validation MSE: {mean_cv_mse}")

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(f'{models_dir}/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model on the entire training data with callbacks
model = build_cnn_model(input_shape, output_shape)
history = model.fit(X_train_poly, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Evaluate on the test data
test_mse = model.evaluate(X_test_poly, y_test, verbose=0)
print(f"Test MSE: {test_mse}")

# Predictions
y_pred = model.predict(X_test_poly)

# Inverse transform the predictions and actual values to get them back to original scale
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"Test R^2 Score: {r2}")

# Save the final model
model.save(f'{models_dir}/trained_model_final.keras')

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(f'{models_dir}/training_history.csv', index=False)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(f'{figures_dir}/training_validation_loss.png')

# Plot predicted vs actual values for each target variable
plt.figure(figsize=(15, 10))
for i in range(y_test.shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
    plt.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], 'r')
    plt.title(f'Target Variable {i+1}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
plt.tight_layout()
plt.savefig(f'{figures_dir}/predicted_vs_actual.png')
