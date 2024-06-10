import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import joblib

# Load the dataset
data = pd.read_csv('data/test_dataset.csv')

# Extract input and output features
X = data[['qubit_frequency_GHz', 'anharmonicity_MHz', 'cavity_frequency_GHz', 'kappa_kHz', 'g_MHz']].values
y = data[['cross_length', 'claw_length', 'EJ', 'coupling_length', 'total_length', 'ground_spacing']].values

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the fitted scaler
joblib.dump(scaler, 'best_dnn_scaler.pkl')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model creation function
def create_model(trial):
    # Suggest hyperparameters for the model
    n_layers = trial.suggest_int('n_layers', 1, 5)
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
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

# Create a study object and specify the direction as 'maximize'
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best trial
best_trial = study.best_trial
print(f"Best trial score: {best_trial.value}")
print("Best hyperparameters: ", best_trial.params)

# Retrieve the best hyperparameters
best_params = best_trial.params

# Create and train the best model using the best hyperparameters
best_model = create_model(trial=optuna.trial.FixedTrial(best_params))
history = best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.2, verbose=1)

# Save the trained model
best_model.save('weights/best_dnn_model.h5')

# Save the best hyperparameters
with open('best_dnn_hyperparameters.json', 'w') as json_file:
    json.dump(best_params, json_file)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('training_validation_loss.pdf')
plt.close()
