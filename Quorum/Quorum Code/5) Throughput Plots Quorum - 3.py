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

# Creating a new column for the seconds part of  'time_received'
client_tx_data['successful'] = (client_tx_data['failed_successful'] >= 0).astype(int)
client_tx_data['failed'] = (client_tx_data['failed_successful'] < 0).astype(int)
client_tx_data = client_tx_data.drop('failed_successful', axis=1)

def plot_message_tx_pool(client_data, output_path):
    """
     Generate and save a scatter plot comparing the response rate against network utilisation for different
     request rates.

     Parameters:
     - txs_received: Filtered DataFrame containing transaction responses.
     - node_network_data: Filtered DataFrame containing network utilisation data of nodes.
     - output_path: The path where the output plot will be saved.
     """
    client_data = client_data[client_data['successful'] == 1]

    # Combine the 'time_sent' and 'time_received' columns into one, with associated labels
    tx_pool = pd.concat([
        client_data[['time_sent', 'freq']].rename(columns={'time_sent': 'time'}).assign(tx=1),
        client_data[['time_received', 'freq']].rename(columns={'time_received': 'time'}).assign(tx=-1)
    ])

    # Sort by the 'time' column
    tx_pool = tx_pool.sort_values(by='time').reset_index(drop=True)
    tx_pool['txs_in_pool'] = tx_pool.groupby('freq')['tx'].cumsum()

    plt.figure(figsize=(8, 6))
    plt.plot(tx_pool.index, tx_pool['txs_in_pool'], label='1', linewidth=3, color = '#6d77aa')

    plt.xlabel('Transactions Sent/Received', size=24)
    plt.ylabel('Transactions in Pool', size=24)
    plt.legend().remove()

    plt.xticks(size=22)
    plt.yticks(size=22)

    # # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)


    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_txs_in_pool.pdf'))
    plt.show()

    return tx_pool


tx_in_pool = plot_message_tx_pool(client_tx_data, figures_directory)


def plot_client_tx_pool(client_data, output_path):
    """
     Generate and save a scatter plot comparing the response rate against network utilisation for different
     request rates.

     Parameters:
     - txs_received: Filtered DataFrame containing transaction responses.
     - node_network_data: Filtered DataFrame containing network utilisation data of nodes.
     - output_path: The path where the output plot will be saved.
     """
    client_data = client_data[client_data['successful'] == 1]
    client_data['sec_sent'] = client_data['time_sent'].dt.floor('S')
    client_data['sec_received'] = client_data['time_received'].dt.floor('S')
    # Drop the original time columns
    client_data = client_data.drop(['time_sent', 'time_received', 'successful', 'failed'], axis=1)
    txs_in = client_data[['sec_sent', 'client']].value_counts().rename('txs_in').to_frame().reset_index()
    txs_out = client_data[['sec_received', 'client']].value_counts().rename('txs_out').to_frame().reset_index()
    txs_in['sec_sent'] = pd.to_datetime(txs_in['sec_sent'])
    txs_out['sec_received'] = pd.to_datetime(txs_out['sec_received'])
    txs_in = txs_in.rename(columns={'sec_sent': 'time'})
    txs_out = txs_out.rename(columns={'sec_received': 'time'})
    txs_in = txs_in.set_index('time')
    txs_out = txs_out.set_index('time')
    txs = pd.merge(txs_in, txs_out, how='outer', on=['time', 'client']).fillna(0).astype(
        {'txs_in': 'int', 'txs_out': 'int'}).reset_index()
    txs = txs.sort_values(by=['time', 'client']).reset_index(drop=True)
    txs['txs_in_pool'] = txs['txs_in'] - txs['txs_out']
    txs = pd.merge(txs, client_data[['sec_sent', 'freq']].drop_duplicates(), how='left', left_on='time',
                   right_on='sec_sent')
    txs = pd.merge(txs, client_data[['sec_received', 'freq']].drop_duplicates(), how='left', left_on='time',
                   right_on='sec_received')
    txs['freq_x'] = txs['freq_x'].fillna(txs['freq_y'])
    txs = txs.drop(['sec_sent', 'sec_received', 'freq_y'], axis=1)
    txs = txs.rename(columns={'freq_x': 'freq'})
    txs['freq'] = txs['freq'].astype(int)
    txs = txs.sort_values(by=['client', 'time'])
    txs['sec'] = txs.groupby(['client', 'freq']).cumcount() + 1
    txs = txs.drop('time', axis=1)
    txs['cum_txs_in_pool'] = txs.groupby(['client', 'freq'])['txs_in_pool'].cumsum()
    mean_txs = txs.groupby(['client', 'freq'])['cum_txs_in_pool'].mean().reset_index()
    pivoted_df_node = mean_txs.pivot(index='freq', columns='client', values='cum_txs_in_pool')

    plt.figure(figsize=(8, 6))

    custom_palette = sns.color_palette("ch:", as_cmap=False, n_colors=(len(pivoted_df_node.columns)))
    sns.set_palette(custom_palette)

    for column in pivoted_df_node.columns:
        plt.plot(pivoted_df_node.index, pivoted_df_node[column], label=f'{column}', linewidth=3)


    plt.xlabel('f$_{req}$', size=24)
    plt.ylabel('Transactions in Pool', size=24)
    plt.legend().remove()

    plt.xticks(size=22)
    plt.yticks(size=22)

    # # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    # Create a color bar with the mapping from colors to client IDs
    cmap = sns.color_palette("ch:", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(pivoted_df_node.columns) - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ticks=[i for i in range(0, len(pivoted_df_node.columns), 2)], spacing='proportional')
    cb.set_label('Client id', size=22)
    cb.ax.tick_params(labelsize=20)

    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_txs_client_in_pool.pdf'))
    plt.show()

    return pivoted_df_node


client_txs_pool = plot_client_tx_pool(client_tx_data, figures_directory)


def plot_rejected_txs(client_data, output_path):
    """
     Generate and save a scatter plot comparing the response rate against network utilisation for different
     request rates.

     Parameters:
     - txs_received: Filtered DataFrame containing transaction responses.
     - node_network_data: Filtered DataFrame containing network utilisation data of nodes.
     - output_path: The path where the output plot will be saved.
     """
    client_data = client_data[client_data['failed'] == 1]
    rejected_txs = client_data.groupby('freq')['failed'].sum().reset_index()

    plt.figure(figsize=(8, 6))
    plt.plot(rejected_txs['freq'], rejected_txs['failed'], label='1', linewidth=3, color = '#6d77aa')


    plt.xlabel('f$_{req}$', size=24)
    plt.ylabel('Rejected transactions', size=24)
    plt.legend().remove()

    plt.xticks(size=22)
    plt.yticks(size=22)

    # # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)


    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_rejected txs.pdf'))
    plt.show()

    return rejected_txs


rejected_txs = plot_rejected_txs(client_tx_data, figures_directory)