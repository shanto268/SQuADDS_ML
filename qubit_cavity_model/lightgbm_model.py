import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import joblib
from .base_model import BaseModel
import matplotlib.pyplot as plt

class LightGBMModel(BaseModel):

    def __init__(self, **model_params):
        super().__init__()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        self.model_qubit = lgb.LGBMRegressor(**model_params)
        self.model_cavity = lgb.LGBMRegressor(**model_params)
        self.poly_features = None

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

        # Ensure original target columns are retained
        df_final = pd.concat([df, df_poly], axis=1)

        return df_final

    def custom_loss_function(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred)**2)
        constraint_violation = np.maximum(
            0, 30 - y_pred[:, 3]).sum()  # Ensure EJEC >= 30
        constraint_violation += np.maximum(
            0, -(y_pred[:, 0] + y_pred[:, 1] + y_pred[:, 2] + y_pred[:, 4] +
                 y_pred[:, 5] - y_pred[:, 3])
        ).sum(
        )  # Ensure `cross_length + cross_gap + ground_spacing + claw_gap + claw_width - claw_length` > 0
        return loss + constraint_violation

    def train(self, df: pd.DataFrame):
        df_final = self.preprocess_data(df, is_training=True)

        # Print columns of df_final for debugging
        print(f"df_final.columns: {df_final.columns}")

        # Make sure to exclude any original features that are duplicated in polynomial features
        X = df_final.loc[:, ~df_final.columns.duplicated()]
        y_qubit = df[self.target_qubit].iloc[:, 0]  # Ensure this is a Series
        y_cavity = df[self.target_cavity].iloc[:, 0]  # Ensure this is a Series

        X_train_qubit, X_test_qubit, y_train_qubit, y_test_qubit = train_test_split(
            X, y_qubit, test_size=0.2, random_state=42)

        X_train_cavity, X_test_cavity, y_train_cavity, y_test_cavity = train_test_split(
            X, y_cavity, test_size=0.2, random_state=42)

        self.model_qubit.fit(X_train_qubit, y_train_qubit)
        self.model_cavity.fit(X_train_cavity, y_train_cavity)

        y_pred_qubit = self.model_qubit.predict(X_test_qubit)
        y_pred_cavity = self.model_cavity.predict(X_test_cavity)

        y_pred_qubit = np.maximum(y_pred_qubit, 0)

        # Ensure y_pred_qubit has at least 4 columns to apply the constraint
        if y_pred_qubit.ndim == 2 and y_pred_qubit.shape[1] > 3:
            y_pred_qubit[:, 3] = np.maximum(y_pred_qubit[:, 3], 30)
            
            constraint_violation_mask = (y_pred_qubit[:, 0] + y_pred_qubit[:, 1] +
                                        y_pred_qubit[:, 2] + y_pred_qubit[:, 4] +
                                        y_pred_qubit[:, 5] - y_pred_qubit[:, 3]
                                        <= 0)
            y_pred_qubit[constraint_violation_mask, :] = np.nan

        r2_qubit = r2_score(y_test_qubit, y_pred_qubit)
        r2_cavity = r2_score(y_test_cavity, y_pred_cavity)

        print(f'R-squared for qubit geometries: {r2_qubit}')
        print(f'R-squared for cavity geometries: {r2_cavity}')

    def hyperparameter_optimization(self, X_train, y_train_qubit, y_train_cavity):
        param_grid = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'max_depth': [-1, 10, 20, 30]
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
        X_poly = pd.DataFrame(self.poly.transform(X_scaled), columns=self.poly.get_feature_names_out(self.features))

        qubit_pred = self.model_qubit.predict(X_poly)
        cavity_pred = self.model_cavity.predict(X_poly)

        qubit_pred = np.maximum(qubit_pred, 0)
        qubit_pred[:, 3] = np.maximum(qubit_pred[:, 3], 30)

        constraint_violation_mask = (qubit_pred[:, 0] + qubit_pred[:, 1] +
                                     qubit_pred[:, 2] + qubit_pred[:, 4] +
                                     qubit_pred[:, 5] - qubit_pred[:, 3] <= 0)
        qubit_pred[constraint_violation_mask, :] = np.nan

        return qubit_pred, cavity_pred


    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['qubit_frequency_Hz'] = df['qubit_frequency_GHz'] * 1e9
        df['cavity_frequency_Hz'] = df['cavity_frequency_GHz'] * 1e9
        df['anharmonicity_Hz'] = df['anharmonicity_MHz'] * 1e6
        df['kappa_Hz'] = df['kappa_kHz'] * 1e3
        df['g_Hz'] = df['g_MHz'] * 1e6

        df_features = df[self.features]
        df_scaled = pd.DataFrame(self.scaler.transform(df_features), columns=self.features)
        df_poly = pd.DataFrame(self.poly.transform(df_scaled), columns=self.poly.get_feature_names_out(self.features))

        # Make sure feature names are unique
        df_poly.columns = [f"poly_{col}" if col in df.columns else col for col in df_poly.columns]

        return df_poly

    def plot_metrics(self, X_test, y_test_qubit, y_test_cavity, model_name, config_name):
        X_test_final = self.preprocess_new_data(X_test)  # Preprocess the test data
        y_pred_qubit = self.model_qubit.predict(X_test_final)
        y_pred_cavity = self.model_cavity.predict(X_test_final)

        # Ensure constraints
        y_pred_qubit = np.maximum(y_pred_qubit, 0)
        y_pred_qubit = np.expand_dims(y_pred_qubit, axis=1)  # Ensure 2D array for consistent indexing
        y_pred_qubit[:, 3] = np.maximum(y_pred_qubit[:, 3], 30)
        constraint_violation_mask = (y_pred_qubit[:, 0] + y_pred_qubit[:, 1] +
                                    y_pred_qubit[:, 2] + y_pred_qubit[:, 4] +
                                    y_pred_qubit[:, 5] - y_pred_qubit[:, 3]
                                    <= 0)
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
        plt.show()
