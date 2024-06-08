import yaml
import os

# Configuration template
config_template = {
    'model': '',
    'dataset_path': 'test_1.csv',
    'new_data_path': 'test_target_params_fig4.csv',
    'qubit_model_path': '',
    'cavity_model_path': '',
    'features': [
        'qubit_frequency_GHz',
        'anharmonicity_MHz',
        'cavity_frequency_GHz',
        'kappa_kHz',
        'g_MHz'
    ],
    'target_qubit': [
        'cross_length',
        'claw_length',
        'EJ'
    ],
    'target_cavity': [
        'coupling_length',
        'total_length'
    ]
}

# Models configuration
models_config = {
    'RandomForest': {
        'qubit_model_path': 'qubit_random_forest_model.pkl',
        'cavity_model_path': 'cavity_random_forest_model.pkl',
        'config_filename': 'random_forest_qubit_cavity.yml'
    },
    'NeuralNetwork': {
        'qubit_model_path': 'qubit_neural_network_model.pkl',
        'cavity_model_path': 'cavity_neural_network_model.pkl',
        'config_filename': 'neural_network_qubit_cavity.yml'
    },
    'XGBoost': {
        'qubit_model_path': 'qubit_xgboost_model.pkl',
        'cavity_model_path': 'cavity_xgboost_model.pkl',
        'config_filename': 'xgboost_qubit_cavity.yml'
    },
    'LightGBM': {
        'qubit_model_path': 'qubit_lightgbm_model.pkl',
        'cavity_model_path': 'cavity_lightgbm_model.pkl',
        'config_filename': 'lightgbm_qubit_cavity.yml'
    },
    'DNN': {
        'qubit_model_path': 'qubit_dnn_model.h5',
        'cavity_model_path': 'cavity_dnn_model.h5',
        'config_filename': 'dnn_qubit_cavity.yml'
    }
}


# Generate configuration files
for model, model_config in models_config.items():
    config = config_template.copy()
    config['model'] = model
    config['qubit_model_path'] = model_config['qubit_model_path']
    config['cavity_model_path'] = model_config['cavity_model_path']
    
    config_filepath = model_config['config_filename']
    with open(config_filepath, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Created config file: {config_filepath}")

print("All config files created successfully.")

