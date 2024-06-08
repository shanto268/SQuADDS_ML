import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import joblib
import tensorflow as tf
from .base_model import BaseModel


class DNNModel(BaseModel):

    def __init__(self, **model_params):
        super().__init__()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        self.model_params = model_params
        self.model_qubit = None
        self.model_cavity = None

    def build_model(self,
                    input_dim,
                    output_dim,
                    layers=3,
                    units=64,
                    dropout_rate=0.2,
                    learning_rate=0.001):
        inputs = Input(shape=(input_dim, ))
        x = Dense(units, activation='relu')(inputs)
        for _ in range(layers - 1):
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
        outputs = Dense(output_dim)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    def preprocess_data(self, df: pd.DataFrame, is_training=True) -> pd.DataFrame:
        df['qubit_frequency_Hz'] = df['qubit_frequency_GHz'] * 1e9
        df['cavity_frequency_Hz'] = df['cavity_frequency_GHz'] * 1e9
        df['anharmonicity_Hz'] = df['anharmonicity_MHz'] * 1e6
        df['kappa_Hz'] = df['kappa_kHz'] * 1e3
        df['g_Hz'] = df['g_MHz'] * 1e6

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

    def custom_loss_function(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
        constraint_violation = tf.reduce_sum(tf.maximum(
            0, 30 - y_pred[:, 3]))  # Ensure EJEC >= 30
        constraint_violation += tf.reduce_sum(
            tf.maximum(
                0, -(y_pred[:, 0] + y_pred[:, 1] + y_pred[:, 2] +
                     y_pred[:, 4] + y_pred[:, 5] - y_pred[:, 3]))
        )  # Ensure `cross_length + cross_gap + ground_spacing + claw_gap + claw_width - claw_length` > 0
        return loss + constraint_violation

    def train(self, df: pd.DataFrame):
        df_final = self.preprocess_data(df, is_training=True)

        X = df_final[self.features]
        y_qubit = df_final[self.target_qubit]
        y_cavity = df_final[self.target_cavity]

        X_train, X_test, y_train_qubit, y_test_qubit = train_test_split(
            X, y_qubit, test_size=0.2, random_state=42)
        X_train, X_test, y_train_cavity, y_test_cavity = train_test_split(
            X, y_cavity, test_size=0.2, random_state=42)

        input_dim = X_train.shape[1]
        output_dim_qubit = y_train_qubit.shape[1]
        output_dim_cavity = y_train_cavity.shape[1]

        self.model_qubit = self.build_model(input_dim, output_dim_qubit,
                                            **self.model_params)
        self.model_cavity = self.build_model(input_dim, output_dim_cavity,
                                             **self.model_params)

        self.model_qubit.fit(X_train,
                             y_train_qubit,
                             epochs=5,
                             batch_size=32,
                             validation_split=0.2)
        self.model_cavity.fit(X_train,
                              y_train_cavity,
                              epochs=5,
                              batch_size=32,
                              validation_split=0.2)

        y_pred_qubit = self.model_qubit.predict(X_test)
        y_pred_cavity = self.model_cavity.predict(X_test)

        y_pred_qubit = np.maximum(y_pred_qubit, 0)

        if y_pred_qubit.shape[1] > 3:
            y_pred_qubit[:, 3] = np.maximum(y_pred_qubit[:, 3], 30)

            constraint_violation_mask = (
                y_pred_qubit[:, 0] + y_pred_qubit[:, 1] + y_pred_qubit[:, 2] +
                y_pred_qubit[:, 4] + y_pred_qubit[:, 5] - y_pred_qubit[:, 3]
                <= 0)
            y_pred_qubit[constraint_violation_mask, :] = np.nan

        r2_qubit = r2_score(y_test_qubit, y_pred_qubit)
        r2_cavity = r2_score(y_test_cavity, y_pred_cavity)

        print(f'R-squared for qubit geometries: {r2_qubit}')
        print(f'R-squared for cavity geometries: {r2_cavity}')

    def hyperparameter_optimization(self, X_train, y_train_qubit,
                                    y_train_cavity):

        def build_keras_model(input_dim,
                              output_dim,
                              layers=3,
                              units=64,
                              dropout_rate=0.2,
                              learning_rate=0.001):
            model = Sequential()
            model.add(Dense(units, input_dim=input_dim, activation='relu'))
            for _ in range(layers - 1):
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(dropout_rate))
            model.add(Dense(output_dim))
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss=self.custom_loss_function)
            return model

        keras_regressor_qubit = KerasRegressor(
            build_fn=build_keras_model,
            input_dim=X_train.shape[1],
            output_dim=y_train_qubit.shape[1],
            verbose=0)
        keras_regressor_cavity = KerasRegressor(
            build_fn=build_keras_model,
            input_dim=X_train.shape[1],
            output_dim=y_train_cavity.shape[1],
            verbose=0)

        param_grid = {
            'layers': [2, 3, 4],
            'units': [64, 128, 256],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01, 0.1],
            'epochs': [50, 100],
            'batch_size': [16, 32, 64]
        }

        random_search_qubit = RandomizedSearchCV(
            estimator=keras_regressor_qubit,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error')
        random_search_qubit.fit(X_train, y_train_qubit)

        random_search_cavity = RandomizedSearchCV(
            estimator=keras_regressor_cavity,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error')
        random_search_cavity.fit(X_train, y_train_cavity)

        self.model_qubit = random_search_qubit.best_estimator_.model
        self.model_cavity = random_search_cavity.best_estimator_.model

        print(
            f'Best parameters for qubit geometries: {random_search_qubit.best_params_}'
        )
        print(
            f'Best parameters for cavity geometries: {random_search_cavity.best_params_}'
        )

    def save_model(self, qubit_model_path: str, cavity_model_path: str):
        self.model_qubit.save(qubit_model_path+ '.keras')
        self.model_cavity.save(cavity_model_path+ '.keras')

    def load_model(self, qubit_model_path: str, cavity_model_path: str):
        self.model_qubit = tf.keras.models.load_model(
            qubit_model_path,
            custom_objects={'custom_loss_function': self.custom_loss_function})
        self.model_cavity = tf.keras.models.load_model(
            cavity_model_path,
            custom_objects={'custom_loss_function': self.custom_loss_function})

    def predict(self, X: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.features]),
                                columns=self.features)
        X_poly = pd.DataFrame(self.poly.transform(X_scaled),
                              columns=self.poly.get_feature_names_out(
                                  self.features))

        qubit_pred = self.model_qubit.predict(X_poly)
        cavity_pred = self.model_cavity.predict(X_poly)

        qubit_pred = np.maximum(qubit_pred, 0)
        if qubit_pred.shape[1] > 3:
            qubit_pred[:, 3] = np.maximum(qubit_pred[:, 3], 30)

            constraint_violation_mask = (qubit_pred[:, 0] + qubit_pred[:, 1] +
                                         qubit_pred[:, 2] + qubit_pred[:, 4] +
                                         qubit_pred[:, 5] - qubit_pred[:, 3]
                                         <= 0)
            qubit_pred[constraint_violation_mask, :] = np.nan

        return qubit_pred, cavity_pred


    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['qubit_frequency_GHz'] = df['qubit_frequency_GHz']
        df['cavity_frequency_GHz'] = df['cavity_frequency_GHz']
        df['anharmonicity_MHz'] = df['anharmonicity_MHz']
        df['kappa_kHz'] = df['kappa_kHz']
        df['g_MHz'] = df['g_MHz']

        df_features = df[['qubit_frequency_GHz', 'anharmonicity_MHz', 'cavity_frequency_GHz', 'kappa_kHz', 'g_MHz']]
        
        print(f"Prediction df_features.columns: {df_features.columns}")
        print(f"Prediction model.features: {self.features}")

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

    def plot_metrics(self, X_test, y_test_qubit, y_test_cavity, model_name, config_name):
        # Ensure the features are scaled and polynomial features are added
        df_scaled = pd.DataFrame(self.scaler.transform(X_test[self.features]), columns=self.features)
        poly_features = self.poly.get_feature_names_out(self.features)
        df_poly = pd.DataFrame(self.poly.transform(df_scaled), columns=poly_features)

        # Combine the original and polynomial features
        X_test_final = pd.concat([X_test.reset_index(drop=True), df_poly], axis=1)

        # Ensure the columns match the expected input for the model
        X_test_final = X_test_final[self.features + list(poly_features)]

        y_pred_qubit = self.model_qubit.predict(X_test_final)
        y_pred_cavity = self.model_cavity.predict(X_test_final)

        # Ensure constraints
        y_pred_qubit = np.maximum(y_pred_qubit, 0)
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
        fig_path = os.path.join('figures', f'{model_name}_{config_name}.png')
        plt.savefig(fig_path)
        plt.show()
