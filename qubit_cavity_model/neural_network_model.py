import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self, **model_params):
        super().__init__()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        self.model_qubit = MLPRegressor(**model_params)
        self.model_cavity = MLPRegressor(**model_params)
    
    def preprocess_data(self, df: pd.DataFrame, is_training=True) -> pd.DataFrame:
        df['qubit_frequency_Hz'] = df['qubit_frequency_GHz'] * 1e9
        df['cavity_frequency_Hz'] = df['cavity_frequency_GHz'] * 1e9
        df['anharmonicity_Hz'] = df['anharmonicity_MHz'] * 1e6
        df['kappa_Hz'] = df['kappa_kHz'] * 1e3
        df['g_Hz'] = df['g_MHz'] * 1e6

        if is_training:
            df = df[df['cross_length'] + df['cross_gap'] + df['ground_spacing'] + df['claw_gap'] + df['claw_width'] - df['claw_length'] > 0]

        df_scaled = pd.DataFrame(self.scaler.fit_transform(df[self.features]), columns=self.features)
        df_poly = pd.DataFrame(self.poly.fit_transform(df_scaled), columns=self.poly.get_feature_names_out(self.features))

        df_final = pd.concat([df, df_poly], axis=1)
        
        return df_final
    
    def custom_loss_function(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred) ** 2)
        constraint_violation = np.maximum(0, 30 - y_pred[:, 3]).sum()  # Ensure EJEC >= 30
        constraint_violation += np.maximum(0, -(
            y_pred[:, 0] + y_pred[:, 1] + y_pred[:, 2] + y_pred[:, 4] + y_pred[:, 5] - y_pred[:, 3]
        )).sum()  # Ensure `cross_length + cross_gap + ground_spacing + claw_gap + claw_width - claw_length` > 0
        return loss + constraint_violation
    
    def train(self, df: pd.DataFrame):
        df_final = self.preprocess_data(df, is_training=True)
        
        X = df_final[self.features]
        y_qubit = df_final[self.target_qubit]
        y_cavity = df_final[self.target_cavity]
        
        X_train, X_test, y_train_qubit, y_test_qubit = train_test_split(X, y_qubit, test_size=0.2, random_state=42)
        X_train, X_test, y_train_cavity, y_test_cavity = train_test_split(X, y_cavity, test_size=0.2, random_state=42)
        
        self.model_qubit.fit(X_train, y_train_qubit)
        self.model_cavity.fit(X_train, y_train_cavity)
        
        y_pred_qubit = self.model_qubit.predict(X_test)
        y_pred_cavity = self.model_cavity.predict(X_test)
        
        y_pred_qubit = np.maximum(y_pred_qubit, 0)
        if y_pred_qubit.shape[1] > 3:
            y_pred_qubit[:, 3] = np.maximum(y_pred_qubit[:, 3], 30)
        
            constraint_violation_mask = (
                y_pred_qubit[:, 0] + y_pred_qubit[:, 1] + y_pred_qubit[:, 2] + y_pred_qubit[:, 4] + y_pred_qubit[:, 5] - y_pred_qubit[:, 3] <= 0
            )
            y_pred_qubit[constraint_violation_mask, :] = np.nan
        
        r2_qubit = r2_score(y_test_qubit, y_pred_qubit)
        r2_cavity = r2_score(y_test_cavity, y_pred_cavity)
        
        print(f'R-squared for qubit geometries: {r2_qubit}')
        print(f'R-squared for cavity geometries: {r2_cavity}')
    
    def hyperparameter_optimization(self, X_train, y_train_qubit, y_train_cavity):
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 100), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }

        grid_search_qubit = GridSearchCV(self.model_qubit, param_grid, cv=5, scoring='r2')
        grid_search_qubit.fit(X_train, y_train_qubit)

        grid_search_cavity = GridSearchCV(self.model_cavity, param_grid, cv=5, scoring='r2')
        grid_search_cavity.fit(X_train, y_train_cavity)

        self.model_qubit = grid_search_qubit.best_estimator_
        self.model_cavity = grid_search_cavity.best_estimator_

        print(f'Best parameters for qubit geometries: {grid_search_qubit.best_params_}')
        print(f'Best parameters for cavity geometries: {grid_search_cavity.best_params_}')

    def save_model(self, qubit_model_path: str, cavity_model_path: str):
        joblib.dump(self.model_qubit, qubit_model_path)
        joblib.dump(self.model_cavity, cavity_model_path)
    
    def load_model(self, qubit_model_path: str, cavity_model_path: str):
        self.model_qubit = joblib.load(qubit_model_path)
        self.model_cavity = joblib.load(cavity_model_path)
    
    def predict(self, X: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.features]), columns=self.features)
        X_poly = pd.DataFrame(self.poly.transform(X_scaled), columns=self.poly.get_feature_names(self.features))
        
        qubit_pred = self.model_qubit.predict(X_poly)
        cavity_pred = self.model_cavity.predict(X_poly)
        
        qubit_pred = np.maximum(qubit_pred, 0)
        qubit_pred[:, 3] = np.maximum(qubit_pred[:, 3], 30)
        
        constraint_violation_mask = (
            qubit_pred[:, 0] + qubit_pred[:, 1] + qubit_pred[:, 2] + qubit_pred[:, 4] + qubit_pred[:, 5] - qubit_pred[:, 3] <= 0
        )
        qubit_pred[constraint_violation_mask, :] = np.nan
        
        return qubit_pred, cavity_pred
