import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, **model_params):
        super().__init__()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        self.model_qubit = xgb.XGBRegressor(**model_params)
        self.model_cavity = xgb.XGBRegressor(**model_params)
    
    def preprocess_data(self, df: pd.DataFrame, is_training=True) -> pd.DataFrame:
        df['qubit_frequency_Hz'] = df['qubit_frequency_GHz'] * 1e9
        df['cavity_frequency_Hz'] = df['cavity_frequency_GHz'] * 1e9
        df['anharmonicity_Hz'] = df['anharmonicity_MHz'] * 1e6
        df['kappa_Hz'] = df['kappa_kHz'] * 1e3
        df['g_Hz'] = df['g_MHz'] * 1e6

        if is_training:
            df = df[df['cross_length'] + df['cross_gap'] + df['ground_spacing'] + df['claw_gap'] + df['claw_width'] - df['claw_length'] > 0]

        df_scaled = pd.DataFrame(self.scaler.fit_transform(df[self.features]), columns=self.features)

        if is_training:
            df_poly = pd.DataFrame(self.poly.fit_transform(df_scaled), columns=self.poly.get_feature_names_out(self.features))
        else:
            df_poly = pd.DataFrame(self.poly.transform(df_scaled), columns=self.poly.get_feature_names_out(self.features))

        # Make sure feature names are unique
        df_poly.columns = [f"poly_{col}" if col in df.columns else col for col in df_poly.columns]

        df_final = pd.concat([df, df_poly], axis=1)
        
        return df_final
    
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
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        grid_search_qubit = GridSearchCV(self.model_qubit, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
        grid_search_qubit.fit(X_train, y_train_qubit)

        grid_search_cavity = GridSearchCV(self.model_cavity, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
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
        poly_features = self.poly.get_feature_names_out(self.features)
        X_poly = pd.DataFrame(self.poly.transform(X_scaled), columns=poly_features)

        # Make sure feature names are unique
        X_poly.columns = [f"poly_{col}" if col in X.columns else col for col in X_poly.columns]
        
        qubit_pred = self.model_qubit.predict(X_poly)
        cavity_pred = self.model_cavity.predict(X_poly)
        
        qubit_pred = np.maximum(qubit_pred, 0)
        if qubit_pred.shape[1] > 3:
            qubit_pred[:, 3] = np.maximum(qubit_pred[:, 3], 30)
        
            constraint_violation_mask = (
                qubit_pred[:, 0] + qubit_pred[:, 1] + qubit_pred[:, 2] + qubit_pred[:, 4] + qubit_pred[:, 5] - qubit_pred[:, 3] <= 0
            )
            qubit_pred[constraint_violation_mask, :] = np.nan
        
        return qubit_pred, cavity_pred
