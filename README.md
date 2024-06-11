# Physics Informed ML Interpolation for SQuADDS


**Note: The documentation below is not updated with information on how to run the simulations on HPC and the models used.**

This project aims to predict the geometrical parameters of a qubit-cavity system given its Hamiltonian parameters using advanced machine learning techniques to replace [SQuADDS](https://arxiv.org/pdf/2312.13483) physics-based interpolation logic. The primary goal is to achieve high prediction accuracy, with an R-squared value greater than 0.9. The project involves training various machine learning models and integrating them into a framework that allows for easy experimentation and real-time predictions.


## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training and Evaluation](#training-and-evaluation)
  - [Prediction](#prediction)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Visualization](#visualization)
- [Contributing](#contributing)

---

## Project Overview

The project is structured to support multiple machine learning models, including:
- Random Forest
- Neural Network (MLP)
- XGBoost
- LightGBM
- Deep Neural Network (DNN)

Each model is designed to predict the following geometrical parameters for qubit and cavity systems:
- Qubit geometries: `cross_length`, `claw_length`, `EJ`
- Cavity geometries: `coupling_length`, `total_length`

The models are trained using features such as `qubit_frequency_GHz`, `anharmonicity_MHz`, `cavity_frequency_GHz`, `kappa_kHz`, and `g_MHz`.

## Features

- **Multiple ML Models**: Easily switch between different models using a configuration file.
- **Hyperparameter Optimization**: Leverage GridSearchCV and RandomizedSearchCV for optimal model tuning.
- **Custom Constraints**: Implemented custom loss functions to handle domain-specific constraints.
- **Scalability**: Designed to scale with larger datasets and high-performance computing (HPC) environments.
- **Visualization**: Plot and save model performance metrics.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shanto268/SQuADDS_ML.git
    cd SQuADDS_ML
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Usage

### Configuration

Create a configuration file (e.g., `config.yaml`) to specify dataset paths, model type, features, and target variables. Below is an example configuration:

```yaml
model: DNN
dataset_path: qubit_cavity_data.csv
new_data_path: new_qubit_cavity_data.csv
qubit_model_path: qubit_model.h5
cavity_model_path: cavity_model.h5
features:
  - qubit_frequency_GHz
  - anharmonicity_MHz
  - cavity_frequency_GHz
  - kappa_kHz
  - g_MHz
target_qubit:
  - cross_length
  - claw_length
  - EJ
target_cavity:
  - coupling_length
  - total_length
```

### Training and Evaluation

To train and evaluate a model, run the `main.py` script with the path to the configuration file:

```bash
python main.py --config config/config.yaml
```

The script will preprocess the data, train the specified model, perform hyperparameter optimization (if enabled), evaluate the model, and save the trained model weights.

#### Disabling Hyperparameter Optimization

If you want to train the model without performing hyperparameter optimization, you can use the `--no_hyper_opt` flag. This can be useful for quickly testing the model training process or if you are confident that the default hyperparameters are sufficient.

```bash
python main.py --config config/config.yaml --no_hyper_opt
```

This flag will skip the hyperparameter optimization step, allowing the model to be trained with the default or previously set parameters.

### Prediction

To make predictions on new data, ensure the trained models are saved and run the `predict.py` script with the new data specified in the configuration file:

```bash
python predict.py --config config/config.yaml
```

The script will load the trained model, preprocess the new data, and generate predictions, which will be saved to the specified output file.

## Hyperparameter Optimization

Hyperparameter optimization is performed using GridSearchCV or RandomizedSearchCV. The optimal parameters are identified based on cross-validation and are printed out after training.

## Visualization

Model performance metrics are plotted and saved in the `figures` directory. The plots include R-squared values and scatter plots of true vs. predicted values.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.
