import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from squadds import SQuADDS_DB, Analyzer
import os
import signal
import sys

# Define the ranges for each parameter
qubit_frequency_range = np.linspace(4, 7, num=10)
anharmonicity_range = np.linspace(-300, -200, num=10)
cavity_frequency_range = np.linspace(6, 10, num=10)
kappa_range = np.linspace(10, 1000, num=10)
g_range = np.linspace(10, 200, num=10)

# Function to generate a DataFrame for each combination of parameters
def generate_target_params_df(qubit_freq, anharm, cavity_freq, kappa, g):
    return pd.DataFrame({
        'qubit_frequency_GHz': [qubit_freq],
        'anharmonicity_MHz': [anharm],
        'cavity_frequency_GHz': [cavity_freq],
        'kappa_kHz': [kappa],
        'g_MHz': [g],
        'res_type': ['quarter'],
        'uid': [f'{qubit_freq}_{anharm}_{cavity_freq}_{kappa}_{g}']
    })

# Function to process each DataFrame and get the merged DataFrame
def process_df(target_params_df):
    analyzer = Analyzer()
    merged_df = analyzer.get_df_with_coupling(target_params_df.iloc[0])

    # Keep only the specified columns
    columns_to_keep = [
        'claw_width', 'cross_length', 'cross_gap', 'claw_length', 'claw_gap',
        'ground_spacing', 'coupler_type', 'resonator_type', 'cavity_frequency_GHz',
        'kappa_kHz', 'EC', 'EJ', 'EJEC', 'qubit_frequency_GHz', 'anharmonicity_MHz',
        'g_MHz', 'coupling_length', 'total_length', 'design_options'
    ]
    df = merged_df.drop(columns=[col for col in merged_df.columns if col not in columns_to_keep])

    # Convert specific columns to float
    float_col_names = ['claw_width', 'claw_length', 'claw_gap']
    for col in float_col_names:
        df[col] = df[col].apply(string_to_float)

    # Apply conversions for specific fields
    with ProcessPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(apply_conversions, [row for _, row in df.iterrows()]))

    # Convert results back to DataFrame
    converted_df = pd.DataFrame(results)

    # Drop unnecessary columns
    converted_df = converted_df.drop(columns=['coupler_type', 'resonator_type', 'design_options'])

    return converted_df

# Function to convert string to float
def string_to_float(string):
    return float(string[:-2])

# Function to apply conversions for specific fields
def apply_conversions(row):
    row['cross_length'] = string_to_float(row['design_options']['qubit_options']['cross_length'])
    row['cross_gap'] = string_to_float(row['design_options']['qubit_options']['cross_gap'])
    row['ground_spacing'] = string_to_float(row['design_options']['qubit_options']['connection_pads']['readout']['ground_spacing'])
    row['coupling_length'] = string_to_float(row['design_options']['cavity_claw_options']['coupler_options']['coupling_length'])
    row['total_length'] = string_to_float(row['design_options']['cavity_claw_options']['cpw_opts']["left_options"]['total_length'])
    return row

# Function to handle the processing and combining of results
def parallel_process_and_combine(dfs, max_workers=64, save_interval=10):
    global combined_merged_df, current_index
    combined_merged_df = pd.DataFrame()
    progress_file = "progress.txt"
    start_index = 0

    # Check if there's a progress file to resume from
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            start_index = int(f.read().strip())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, merged_df in enumerate(executor.map(process_df, dfs[start_index:]), start=start_index):
            combined_merged_df = pd.concat([combined_merged_df, merged_df], ignore_index=True)
            current_index = i

            # Save progress at intervals
            if (i + 1) % save_interval == 0:
                save_progress(combined_merged_df, i + 1)

    # Save the final results
    save_progress(combined_merged_df, len(dfs))

    return combined_merged_df

def save_progress(df, index):
    df.to_csv("quarter_wave_dataset.csv", index=False)
    df.to_parquet("quarter_wave_dataset.parquet", index=False)
    with open("progress.txt", "w") as f:
        f.write(str(index))

def signal_handler(sig, frame):
    save_progress(combined_merged_df, current_index)
    sys.exit(0)

# Register the signal handler for multiple termination signals
signal.signal(signal.SIGINT, signal_handler)   # Interrupt from keyboard (Ctrl+C)
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
signal.signal(signal.SIGHUP, signal_handler)   # Hangup detected on controlling terminal
signal.signal(signal.SIGQUIT, signal_handler)  # Quit from keyboard

# Define the ranges for each parameter
qubit_frequency_range = np.linspace(4, 7, num=10)
anharmonicity_range = np.linspace(-300, -200, num=10)
cavity_frequency_range = np.linspace(6, 10, num=10)
kappa_range = np.linspace(10, 1000, num=10)
g_range = np.linspace(10, 200, num=10)

print("Establishing DB connection....")
db = SQuADDS_DB()
db.select_system(["qubit", "cavity_claw"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("quarter")
df = db.create_system_df()
print("DB Connection Established!")

# Prepare a list of DataFrames for each combination of parameters
target_params_dfs = [
    generate_target_params_df(qf, a, cf, k, g)
    for qf in qubit_frequency_range
    for a in anharmonicity_range
    for cf in cavity_frequency_range
    for k in kappa_range
    for g in g_range
]

# Process all DataFrames in parallel and combine them
combined_merged_df = parallel_process_and_combine(target_params_dfs, max_workers=64)
print("Processing done!")

# Save the final results
combined_merged_df.to_csv("quarter_wave_dataset.csv", index=False)
combined_merged_df.to_parquet("quarter_wave_dataset.parquet", index=False)
