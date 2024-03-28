import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os

# Global constants for directory paths to organize data and figures
data_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/DLPS - Quorum Raw Data'
figures_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/Figures/Throughput/'

# General settings for the analysis
client_number = 16
node_number = 8
instance_memory = 67108864
number_cpus = 16
frequencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
               1900, 2000, 2100, 2200, 2300, 2400, 2500]

# Define the time frame for analysis
start_seconds = 3
end_seconds = 17


def ensure_dir(directory):
    """
    Ensure that the specified directory exists, creating it if necessary.

    Parameters:
    - directory: The path of the directory to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_single_file_tx_data(directory_path, freq_number, file_type, identifier):
    """
    Loads a single CSV file into a pandas DataFrame.

    Parameters:
    - directory_path: The path to the directory containing the raw data files.
    - freq_number: The frequency number associated with the file.
    - file_type: The type of the node (client, peer, orderer).
    - identifier: The identifier number for the specific node.

    Returns:
    - A DataFrame containing the data from the file
    """
    file_path = f'{directory_path}/freq{freq_number}_{file_type}{identifier}_tx_data.csv'

    # Generate column names and read file
    data = pd.read_csv(file_path, sep=' ', header=None).iloc[:-1, :]
    data.columns = ['time_sent'] + ['time_received'] + ['failed_successful']
    data['time_sent'] = data['time_sent'].astype(float)
    data['time_received'] = data['time_received'].astype(float)

    # Convert Unix time to datetime and calculate differences
    for col in ['time_sent', 'time_received']:
        data[col] = pd.to_datetime(data[col], unit='ms')

    data[file_type] = f'{file_type}{identifier}'  # Add file_type name (client/node) as a new column
    data['freq'] = freq_number  # Add frequency as a new column

    return data


def load_data_for_frequencies(frequencies, file_type, directory_path):
    """
    Load and concatenate data for all specified frequencies and node type.

    Parameters:
    - frequencies: List of frequency values to load data for.
    - file_type: Type of node (client, node).
    - directory_path: Base directory where data files are stored.

    Returns:
    - A DataFrame containing all the loaded and concatenated data.
    """
    all_dfs = []
    for freq in frequencies:
        dfs_for_current_freq = []

        for i in range(client_number):
            dfs_for_current_freq.append(load_single_file_tx_data(directory_path, freq, file_type, i))
        all_dfs.extend(dfs_for_current_freq)

    return pd.concat(all_dfs, ignore_index=True)


client_tx_data = load_data_for_frequencies(frequencies, 'client', data_directory)
client_tx_data = client_tx_data.sort_values(by=['freq', 'client', 'time_sent'])

# Creating a new column for the seconds part of 'time_sent' and 'time_received'
client_tx_data['received_second'] = client_tx_data['time_received'].dt.floor('S')
client_tx_data['successful'] = (client_tx_data['failed_successful'] >= 0).astype(int)
client_tx_data['failed'] = (client_tx_data['failed_successful'] < 0).astype(int)
client_tx_data = client_tx_data.drop('failed_successful', axis=1)

# Calculating the number of transactions processed per second
txs_received_per_second = client_tx_data[client_tx_data['successful']==1]
txs_received_per_second = txs_received_per_second.groupby(['received_second', 'freq']).size().reset_index(name='txs_received')
txs_received_per_second['secs'] = txs_received_per_second.groupby('freq').cumcount() + 1
txs_received_per_second = txs_received_per_second[(txs_received_per_second['secs'] >= start_seconds) &
                                                  (txs_received_per_second['secs'] <= end_seconds)]

# Set 'received_second' as the index
txs_received_per_second.set_index('received_second', inplace=True)
# Calculate values for windows
txs_received_per_second['window_1s'] = txs_received_per_second['txs_received'].rolling('1s').sum()
txs_received_per_second['window_3s'] = txs_received_per_second['txs_received'].rolling('3s').sum()/3
txs_received_per_second.loc[txs_received_per_second['secs'].between(3, 4), 'window_3s'] = float('nan')
txs_received_per_second['window_8s'] = txs_received_per_second['txs_received'].rolling('8s').sum()/8
txs_received_per_second.loc[txs_received_per_second['secs'].between(3, 9), 'window_8s'] = float('nan')
# Per frequency
txs_received_mean_per_freq = txs_received_per_second.groupby('freq')['txs_received'].mean().reset_index()


def tx_request_response(txs_received, x_col, y_col, output_path):
    """
    Generate and save a scatter plot comparing tx response against the request rate.

    Parameters:
    - txs_received: DataFrame containing the response rate of the txs
    - x_col: Column containing the request rate.
    - y_col: Column containing the response rate.
    - output_path: The path where the output plot will be saved.
    """
    plt.figure(figsize=(8, 6))

    sns.scatterplot(data=txs_received, x=x_col, y=y_col, s=50, alpha=0.6, color='#6d77aa')

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('f$_{resp}$', size=26)

    plt.xticks(size=22)
    plt.yticks(size=22)
    plt.ylim(0, 2500)
    plt.legend().remove()

    # # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    # # Hide every second tick
    # tick_labels = plt.yticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, f'q_tx_request_response_{y_col}.pdf'))

    plt.show()


tx_request_response(txs_received_per_second, 'freq', 'window_1s', figures_directory)
tx_request_response(txs_received_per_second, 'freq', 'window_3s', figures_directory)
tx_request_response(txs_received_per_second, 'freq', 'window_8s', figures_directory)
tx_request_response(txs_received_mean_per_freq, 'freq', 'txs_received', figures_directory)
