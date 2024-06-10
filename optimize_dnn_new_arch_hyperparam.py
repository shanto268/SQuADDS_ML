import sys
from datetime import datetime

import joblib
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Generate a unique identifier with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fname_id = sys.argv[0].split('.')[0] + '_' + timestamp

# Load data
training_data = pd.read_csv("data/test_dataset_2.csv")

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
joblib.dump(poly, f'models/poly_transformer_{fname_id}.pkl')

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_poly = scaler_X.fit_transform(X_train_poly)
X_test_poly = scaler_X.transform(X_test_poly)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Save the scalers
joblib.dump(scaler_X, f'models/scaler_X_{fname_id}.pkl')
joblib.dump(scaler_y, f'models/scaler_y_{fname_id}.pkl')


# Define the model creation function
def create_model(trial):
    neurons1 = trial.suggest_int('neurons1', 128, 512)
    neurons2 = trial.suggest_int('neurons2', 64, 256)
    neurons3 = trial.suggest_int('neurons3', 32, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    model = Sequential()
    model.add(Input(shape=(X_train_poly.shape[1],)))
    model.add(Dense(neurons1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(neurons2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(neurons3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(y_train.shape[1]))  # Output layer with the same number of neurons as output features
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Define the objective function for Optuna
def objective(trial):
    model = create_model(trial)
    
    history = model.fit(X_train_poly, y_train, 
                        validation_split=0.2,
                        epochs=50, 
                        batch_size=32, 
                        verbose=0)
    
    # Evaluate the model
    loss = model.evaluate(X_test_poly, y_test, verbose=0)
    
    # Save the model if it has the best performance so far
    if trial.should_prune():
        raise optuna.TrialPruned()

    return loss

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Save the best hyperparameters
joblib.dump(study.best_params, f'models/best_hyperparameters_{fname_id}.pkl')

# Save the best model
best_model = create_model(optuna.trial.FixedTrial(study.best_params))
best_model.fit(X_train_poly, y_train, epochs=50, batch_size=32, verbose=0)
best_model.save(f'models/best_model_{fname_id}.keras')

# Output the best hyperparameters
print("Best hyperparameters: ", study.best_params)
