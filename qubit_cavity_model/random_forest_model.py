import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .base_model import BaseModel

# Setup logging
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(filename=f'logs/training_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RandomForestModel(BaseModel):
    def __init__(self, qubit_params=None, cavity_params=None):
        super().__init__()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        if qubit_params is None:
            qubit_params = {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
        if cavity_params is None:
            cavity_params = {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
        self.model_qubit = RandomForestRegressor(**qubit_params)
        self.model_cavity = RandomForestRegressor(**cavity_params)
        self.poly_features = None
    
    def preprocess_data(self, df: pd.DataFrame, is_training=True) -> pd.DataFrame:
        """
        df['qubit_frequency_GHz'] = df['qubit_frequency_GHz']
        df['cavity_frequency_GHz'] = df['cavity_frequency_GHz']
        df['anharmonicity_MHz'] = df['anharmonicity_MHz']
        df['kappa_kHz'] = df['kappa_kHz']
        df['g_MHz'] = df['g_MHz']
        """

        if is_training:
            df = df[df['cross_length'] + df['cross_gap'] + df['ground_spacing'] + df['claw_gap'] + df['claw_width'] - df['claw_length'] > 0]

        df_scaled = pd.DataFrame(self.scaler.fit_transform(df[self.features]), columns=self.features)
        self.poly.fit(df_scaled)  # Fit the polynomial features
        poly_features = self.poly.get_feature_names_out(self.features)
        df_poly = pd.DataFrame(self.poly.transform(df_scaled), columns=poly_features)

        # Make sure feature names are unique
        df_poly.columns = [f"poly_{col}" if col in df.columns else col for col in df_poly.columns]
        self.poly_features = df_poly.columns

        df_final = pd.concat([df, df_poly], axis=1)
        
        return df_final

        
    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert the columns in the new data to match the training feature names
        df['qubit_frequency_GHz'] = df['qubit_frequency_GHz']
        df['cavity_frequency_GHz'] = df['cavity_frequency_GHz']
        df['anharmonicity_MHz'] = df['anharmonicity_MHz']
        df['kappa_kHz'] = df['kappa_kHz']
        df['g_MHz'] = df['g_MHz']

        df_features = df[['qubit_frequency_GHz', 'anharmonicity_MHz', 'cavity_frequency_GHz', 'kappa_kHz', 'g_MHz']]
        
        # Ensure the feature names in df_features match the feature names in model.features
        df_features.columns = self.features

        # Transform the features using the same scaler and polynomial features from training
        df_scaled = pd.DataFrame(self.scaler.transform(df_features), columns=self.features)
        poly_features = self.poly.get_feature_names_out(self.features)
        df_poly = pd.DataFrame(self.poly.transform(df_scaled), columns=poly_features)

        # Make sure feature names are unique
        df_poly.columns = [f"poly_{col}" if col in df.columns else col for col in df_poly.columns]

        df_final = pd.concat([df, df_poly], axis=1)

        return df_final

    
    def train(self, df: pd.DataFrame):
        df_final = self.preprocess_data(df, is_training=True)
        print(df_final.head())
        print(df_final.dtypes)
        
        X = df_final[self.features + list(self.poly_features)]
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
        
        logging.info(f'R-squared for qubit geometries: {r2_qubit}')
        logging.info(f'R-squared for cavity geometries: {r2_cavity}')
    
    def hyperparameter_optimization(self, X_train, y_train_qubit, y_train_cavity):
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        grid_search_qubit = GridSearchCV(self.model_qubit, param_grid, cv=5, scoring='r2')
        grid_search_qubit.fit(X_train, y_train_qubit)

        grid_search_cavity = GridSearchCV(self.model_cavity, param_grid, cv=5, scoring='r2')
        grid_search_cavity.fit(X_train, y_train_cavity)

        self.model_qubit = grid_search_qubit.best_estimator_
        self.model_cavity = grid_search_cavity.best_estimator_

        logging.info(f'Best parameters for qubit geometries: {grid_search_qubit.best_params_}')
        logging.info(f'Best parameters for cavity geometries: {grid_search_cavity.best_params_}')

    def save_model(self, qubit_model_path: str, cavity_model_path: str):
        joblib.dump(self.model_qubit, qubit_model_path)
        joblib.dump(self.model_cavity, cavity_model_path)
        joblib.dump(self.scaler, qubit_model_path.replace(".pkl", "_scaler.pkl"))
        joblib.dump(self.poly, qubit_model_path.replace(".pkl", "_poly.pkl"))
    
    def load_model(self, qubit_model_path: str, cavity_model_path: str):
        self.model_qubit = joblib.load(qubit_model_path)
        self.model_cavity = joblib.load(cavity_model_path)
        self.scaler = joblib.load(qubit_model_path.replace(".pkl", "_scaler.pkl"))
        self.poly = joblib.load(qubit_model_path.replace(".pkl", "_poly.pkl"))
        # Ensure features are set
        self.features = ['qubit_frequency_Hz', 'cavity_frequency_Hz', 'anharmonicity_Hz', 'kappa_Hz', 'g_Hz']
    
    def predict(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        df_final = self.preprocess_new_data(df)
        X_poly = df_final[self.features + list(self.poly.get_feature_names_out(self.features))]
        
        qubit_pred = self.model_qubit.predict(X_poly)
        cavity_pred = self.model_cavity.predict(X_poly)
        
        qubit_pred = np.maximum(qubit_pred, 0)
        qubit_pred[:, 3] = np.maximum(qubit_pred[:, 3], 30)
        
        constraint_violation_mask = (
            qubit_pred[:, 0] + qubit_pred[:, 1] + qubit_pred[:, 2] + qubit_pred[:, 4] + qubit_pred[:, 5] - qubit_pred[:, 3] <= 0
        )
        qubit_pred[constraint_violation_mask, :] = np.nan
        
        return pd.DataFrame(qubit_pred), pd.DataFrame(cavity_pred)
