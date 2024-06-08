import argparse
import os
import pandas as pd
import yaml
from qubit_cavity_model.random_forest_model import RandomForestModel
from qubit_cavity_model.neural_network_model import NeuralNetworkModel
from qubit_cavity_model.dnn_model import DNNModel
from qubit_cavity_model.xgboost_model import XGBoostModel

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
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

def main(config_path):
    config = load_config(config_path)
    model_name = config['model']
    config_name = os.path.basename(config_path).split('.')[0]

    # Initialize the model
    model = get_model(config)
    print(f"Model Initialized: {model_name}")

    # Load the model
    model.load_model(f"weights/{config['qubit_model_path']}", f"weights/{config['cavity_model_path']}")

    # Load new data
    print(f"Predicting on new data: {config['new_data_path']}")
    new_data = pd.read_csv(f"data/{config['new_data_path']}")

    # Predict on new data
    qubit_pred, cavity_pred = model.predict(new_data)
    print(f"Predictions done.\nQubit Predictions: {qubit_pred.head()}\nCavity Predictions: {cavity_pred.head()}")

    # Save the predicted data to a CSV file
    qubit_pred.to_csv(f"predictions/{config_name}_qubit_predictions.csv", index=False)
    cavity_pred.to_csv(f"predictions/{config_name}_cavity_predictions.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict qubit-cavity models.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')

    args = parser.parse_args()
    main(args.config)
