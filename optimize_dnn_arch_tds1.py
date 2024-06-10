import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('data/test_dataset.csv')

# Extract input and output features
X = data[['qubit_frequency_GHz', 'anharmonicity_MHz', 'cavity_frequency_GHz', 'kappa_kHz', 'g_MHz']].values
y = data[['cross_length', 'claw_length', 'EJ', 'coupling_length', 'total_length', 'ground_spacing']].values

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the fitted scaler
joblib.dump(scaler, 'weights/best_dnn_scaler_optimized_test_dataset_1.pkl')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert the datasets to tf.data.Dataset format
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Function to preprocess the data
def preprocess(X, y):
    return X, y

# Apply preprocessing in parallel
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

# Shuffle and batch the datasets
BATCH_SIZE = 32

train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Define the model creation function
def create_model(trial):
    # Suggest hyperparameters for the model
    n_layers = trial.suggest_int('n_layers', 1, 7)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    loss = trial.suggest_categorical('loss', ['mean_squared_error', 'mean_absolute_error', 'huber'])

    model = Sequential()
    input_dim = X_train.shape[1]
    
    for i in range(n_layers):
        num_neurons = trial.suggest_int(f'n_units_l{i}', 16, 256)
        if i == 0:
            model.add(Input(shape=(input_dim,)))
            model.add(Dense(num_neurons, activation=activation))
        else:
            model.add(Dense(num_neurons, activation=activation))
    
    model.add(Dense(y_train.shape[1]))  # Output layer with the same number of neurons as output features
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    return model

# Define the objective function for Optuna
def objective(trial):
    model = create_model(trial)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    epochs = trial.suggest_int('epochs', 50, 200)
    
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=0)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

# Create a study object and specify the direction as 'maximize'
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# Get the best trial
best_trial = study.best_trial
print(f"Best trial score: {best_trial.value}")
print("Best hyperparameters: ", best_trial.params)

# Retrieve the best hyperparameters
best_params = best_trial.params

# Create and train the best model using the best hyperparameters
best_model = create_model(trial=optuna.trial.FixedTrial(best_params))
history = best_model.fit(train_dataset, epochs=best_params['epochs'], validation_data=test_dataset, batch_size=best_params['batch_size'], verbose=1)

# Save the trained model
best_model.save('weights/best_dnn_model_optimized_test_dataset_1.h5')

# Save the best hyperparameters
with open('best_dnn_hyperparameters_optimized_test_dataset_1.json', 'w') as json_file:
    json.dump(best_params, json_file)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('training_validation_loss_optimized_test_dataset_1.pdf')
plt.close()
