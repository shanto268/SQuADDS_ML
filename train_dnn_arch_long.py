import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load data
training_data = pd.read_csv(f"data/test_dataset_2.csv")

# Extract input and output features
X = training_data[['qubit_frequency_GHz', 'anharmonicity_MHz', 'cavity_frequency_GHz', 'kappa_kHz', 'g_MHz']].values
y = training_data[['cross_length', 'claw_length', 'EJ', 'coupling_length', 'total_length', 'ground_spacing']].values

# Split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example of creating polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Save the polynomial feature transformer
joblib.dump(poly, 'models/poly_transformer.pkl')

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_poly = scaler_X.fit_transform(X_train_poly)
X_test_poly = scaler_X.transform(X_test_poly)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Save the scalers
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

# Define the model creation function
def create_model(neurons1=256, neurons2=128, neurons3=64, learning_rate=0.001, trial=None):
    if trial:
        # Suggest hyperparameters for the model if trial is provided
        neurons1 = trial.suggest_int('neurons1', 128, 512)
        neurons2 = trial.suggest_int('neurons2', 64, 256)
        neurons3 = trial.suggest_int('neurons3', 32, 128)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    model = Sequential()
    model.add(Dense(neurons1, input_dim=X_train_poly.shape[1], activation='relu'))
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
mean_cv_mse = cross_val_score_keras(lambda: create_model(), X_train_poly, y_train, cv=2, epochs=100, batch_size=32)
print(f"Mean Cross-Validation MSE: {mean_cv_mse}")

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model on the entire training data with callbacks
model = create_model()
history = model.fit(X_train_poly, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate on the test data
test_mse = model.evaluate(X_test_poly, y_test, verbose=0)
print(f"Test MSE: {test_mse}")

# Predictions
y_pred = model.predict(X_test_poly)

# Inverse transform the predictions and actual values to get them back to original scale
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# Calculate R^2 score
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"Test R^2 Score: {r2}")

# Save the final model
model.save('models/trained_model_final.h5')

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('models/training_history.csv', index=False)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('figures/training_validation_loss.png')

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
plt.savefig('figures/predicted_vs_actual.png')
