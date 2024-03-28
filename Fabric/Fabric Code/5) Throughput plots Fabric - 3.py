import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os

# Global constants for directory paths to organize data and figures
data_directory = 'What Blocks My Blockchain’s Throughput - Data/Fabric/DLPS - Fabric Raw Data/'
figures_directory = 'What Blocks My Blockchain’s Throughput - Data/Fabric/Figures/Throughput/'

# General settings for the analysis
client_number = 16
peer_number = 8
orderer_number = 4
instance_memory = 67108864
number_cpus = 16
frequencies = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]


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

    data[file_type] = f'{file_type}{identifier}'  # Add file_type name (client/orderer/peer) as a new column
    data['freq'] = freq_number  # Add frequency as a new column

    return data


def load_data_for_frequencies(frequencies, file_type, directory_path):
    """
    Load and concatenate data for all specified frequencies and node type.

    Parameters:
    - frequencies: List of frequency values to load data for.
    - file_type: Type of node (client, peer, orderer).
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

# Creating a new column for the seconds part of  'time_received'
client_tx_data['received_second'] = client_tx_data['time_received'].dt.floor('S')
client_tx_data['successful'] = (client_tx_data['failed_successful'] >= 0).astype(int)
client_tx_data['failed'] = (client_tx_data['failed_successful'] < 0).astype(int)
client_tx_data = client_tx_data.drop('failed_successful', axis=1)

# Calculating the number of transactions processed per second
txs_received_per_second = client_tx_data[client_tx_data['successful']==1]
txs_received_per_second = txs_received_per_second.groupby(['received_second', 'client', 'freq']).size().reset_index(name='txs_received')
txs_received_per_second['secs'] = txs_received_per_second.groupby(['freq', 'client']).cumcount() + 1
txs_received_per_second = txs_received_per_second[(txs_received_per_second['secs'] >= 3) &
                                                  (txs_received_per_second['secs'] <= 17)]

txs_received_per_second.set_index('received_second', inplace=True)
txs_received_per_second = txs_received_per_second.groupby(['client', 'freq']).rolling(window=3, min_periods=1)['txs_received'].sum()/3
txs_received_per_second = txs_received_per_second.reset_index()
txs_received_per_second = txs_received_per_second.groupby(['client', 'freq']).apply(lambda x: x.iloc[2:]).reset_index(drop=True)
txs_received_per_second['secs'] = txs_received_per_second.groupby(['client', 'freq']).cumcount()
txs_received_per_second = txs_received_per_second.drop('received_second', axis=1)

client_to_peer = {}
for i in range(0, 16):
    client_to_peer[f'client{i}'] = f'peer{int(i / 2)}'

txs_received_per_second['peer'] = txs_received_per_second['client'].map(client_to_peer)
txs_received_per_second = txs_received_per_second.drop('client', axis=1)
txs_received_per_second = txs_received_per_second.groupby(['peer', 'freq', 'secs']).agg({'txs_received': 'sum'}).reset_index()


def mean_response_peer_plot(txs_received, output_path):
    """
    Generates and saves a line plot showing the average CPU usage for each peer across different frequencies.

    Parameters:
    - peer_data_15s: Filtered DataFrame containing data for peers in a 15 sec window.
    - output_path: The directory path where the output plot will be saved.
    """
    peer_freq_means = txs_received.groupby(['peer', 'freq']).agg({'txs_received': 'mean'}).reset_index()
    peer_freq_means['peer_id'] = peer_freq_means['peer'].str.extract('(\d+)').astype(int)
    pivoted_df = peer_freq_means.pivot(index='freq', columns='peer_id', values='txs_received')

    # Create a custom color palette
    custom_palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=(len(pivoted_df.columns)))
    sns.set_palette(custom_palette)

    # Create the line plot
    plt.figure(figsize=(8, 6))

    for column in pivoted_df.columns:
        sns.lineplot(x=pivoted_df.index, y=pivoted_df[column], label=f'{column}', linewidth=3)

    plt.legend(title='Peer id', fontsize=20, title_fontsize=20)

    plt.xticks(size=22)
    plt.yticks(size=22)

    # Hide every second tick
    tick_labels = plt.xticks()[1]
    for i, label in enumerate(tick_labels):
        if i % 2 == 1:
            label.set_visible(False)

    plt.xlabel('$f_{req}$', size=26)
    plt.ylabel('$f_{resp}$', size=24)
    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'mean_response_peer_plot.pdf'))

    plt.show()


mean_response_peer_plot(txs_received_per_second, figures_directory)

