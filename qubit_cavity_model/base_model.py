import os
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import logging
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class BaseModel(ABC):
    def __init__(self):
        self.features = None
        self.target_qubit = None
        self.target_cavity = None
        self.scaler = None
        self.poly = None
        self.poly_features = None

    def set_config(self, config):
        self.features = config['features']
        self.target_qubit = config['target_qubit']
        self.target_cavity = config['target_cavity']


    @abstractmethod
    def train(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def save_model(self, qubit_model_path: str, cavity_model_path: str):
        pass

    @abstractmethod
    def load_model(self, qubit_model_path: str, cavity_model_path: str):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        pass

    @abstractmethod
    def hyperparameter_optimization(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def plot_metrics(self, X_test, y_test_qubit, y_test_cavity, model_name, config_name):
        y_pred_qubit = self.model_qubit.predict(X_test)
        y_pred_cavity = self.model_cavity.predict(X_test)

        # Ensure constraints
        y_pred_qubit = np.maximum(y_pred_qubit, 0)

        if y_pred_qubit.shape[1] > 3:
            y_pred_qubit[:, 3] = np.maximum(y_pred_qubit[:, 3], 30)
            constraint_violation_mask = (
                y_pred_qubit[:, 0] + y_pred_qubit[:, 1] + y_pred_qubit[:, 2] + y_pred_qubit[:, 4] + y_pred_qubit[:, 5] - y_pred_qubit[:, 3] <= 0
            )
            y_pred_qubit[constraint_violation_mask, :] = np.nan

        r2_qubit = r2_score(y_test_qubit, y_pred_qubit)
        r2_cavity = r2_score(y_test_cavity, y_pred_cavity)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y_test_qubit.values.flatten(), y_pred_qubit.flatten())
        axes[0].set_title(f'Qubit R-squared: {r2_qubit}')
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        
        axes[1].scatter(y_test_cavity.values.flatten(), y_pred_cavity.flatten())
        axes[1].set_title(f'Cavity R-squared: {r2_cavity}')
        axes[1].set_xlabel('True Values')
        axes[1].set_ylabel('Predicted Values')
        
        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        fig_path = os.path.join('figures', f'{model_name}_{config_name}_{timestamp}.png')
        plt.savefig(fig_path)
