import argparse
import os

import pandas as pd
import yaml
import logging
from sklearn.model_selection import train_test_split

from qubit_cavity_model.dnn_model import DNNModel
from qubit_cavity_model.neural_network_model import NeuralNetworkModel
from qubit_cavity_model.random_forest_model import RandomForestModel
from qubit_cavity_model.xgboost_model import XGBoostModel
from qubit_cavity_model.lightgbm_model import LightGBMModel
from datetime import datetime

# Setup logging
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(filename=f'logs/training_{timestamp}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_tree(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")
        else:
            print(f"Directory {directory} already exists.")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model(config):
    model_name = config['model']
    if model_name == 'RandomForest':
        return RandomForestModel()
    elif model_name == 'NeuralNetwork':
        return NeuralNetworkModel()
    elif model_name == 'DNN':
        return DNNModel()
    elif model_name == 'XGBoost':
        return XGBoostModel()
    elif model_name == 'LightGBM':
        return LightGBMModel()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

def main(config_path, no_hyper_opt):
    directories = ['predictions', 'weights', 'logs', 'figures']
    create_tree(directories)

    config = load_config(config_path)
    config_name = os.path.basename(config_path).split('.')[0]
    model_name = config['model']

    # Load your dataset
    df = pd.read_csv(f"data/{config['dataset_path']}")
    print(f"Data Loaded.\n Dataframe: {df.head()}")

    # Initialize the model
    model = get_model(config)
    model.set_config(config)
    print(f"Model Initialized: {model_name}")

    # Train the model
    print("Training the model...")
    model.train(df)
    print("Model trained.")

    df_final = model.preprocess_data(df, is_training=True)

    if model_name == 'LightGBM':
        X = df_final[model.features + list(model.poly.get_feature_names_out(model.features))]

        y_qubit = df_final[model.target_qubit].values[:, 0]  # Ensure single column
        y_cavity = df_final[model.target_cavity].values[:, 0]  # Ensure single column

        # Ensure there are no duplicate columns
        X = X.loc[:,~X.columns.duplicated()]
    else:
        try:
            X = df_final[model.features]
        except:
            X = df_final[model.features + list(model.poly_features)]

        y_qubit = df_final[model.target_qubit]
        y_cavity = df_final[model.target_cavity]

    X_train, X_test, y_train_qubit, y_test_qubit = train_test_split(X, y_qubit, test_size=0.2, random_state=42)
    X_train, X_test, y_train_cavity, y_test_cavity = train_test_split(X, y_cavity, test_size=0.2, random_state=42)

    if not no_hyper_opt:
        # Hyperparameter optimization
        print("Hyperparameter optimization...")
        model.hyperparameter_optimization(X_train, y_train_qubit, y_train_cavity)
        print("Hyperparameter optimization done.")
        # Save the best optimized models
        model.save_model(f"weights/{config['qubit_model_path'].replace('.pkl', '_optimized.pkl')}", 
                         f"weights/{config['cavity_model_path'].replace('.pkl', '_optimized.pkl')}")

    # Save the model
    model.save_model(f"weights/{config['qubit_model_path']}", f"weights/{config['cavity_model_path']}")
    print("Model saved.")

    
    if model_name == 'LightGBM':
        # X_test_final = model.preprocess_new_data(X_test)
        raise NotImplementedError("LightGBM model does not support plotting metrics.")
        # model.plot_metrics(X_test_final, y_test_qubit, y_test_cavity, model_name, config_name)
    else:
        model.plot_metrics(X_test, y_test_qubit, y_test_cavity, model_name, config_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train qubit-cavity models.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--no_hyper_opt', action='store_true', help='Disable hyperparameter optimization.')

    args = parser.parse_args()
    main(args.config, args.no_hyper_opt)
