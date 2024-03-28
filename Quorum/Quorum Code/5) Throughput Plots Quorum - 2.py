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
client_tx_data['received_second'] = client_tx_data['time_received'].dt.floor('S')
client_tx_data['successful'] = (client_tx_data['failed_successful'] >= 0).astype(int)
client_tx_data['failed'] = (client_tx_data['failed_successful'] < 0).astype(int)
client_tx_data = client_tx_data.drop('failed_successful', axis=1)

# Calculating the number of transactions processed per second
txs_received_per_second = client_tx_data[client_tx_data['successful']==1]
txs_received_per_second = txs_received_per_second.groupby(['received_second', 'freq']).size().reset_index(name='txs_received')
txs_received_per_second['secs'] = txs_received_per_second.groupby('freq').cumcount() + 1
txs_received_per_second = txs_received_per_second[(txs_received_per_second['secs'] >= 3) &
                                                  (txs_received_per_second['secs'] <= 17)]

txs_received_per_second.set_index('received_second', inplace=True)
txs_received_per_second['window_3s'] = txs_received_per_second['txs_received'].rolling('3s').sum()/3
txs_received_per_second.loc[txs_received_per_second['secs'].between(3, 4), 'window_3s'] = float('nan')
txs_received_per_second = txs_received_per_second.reset_index()
txs_received_per_second['time'] = pd.to_datetime(txs_received_per_second['received_second'], format='%H:%M:%S').dt.time
txs_received_per_second = txs_received_per_second[['time', 'secs', 'freq', 'window_3s']].dropna()


def load_single_file(directory_path, freq_number, file_type, identifier):
    """
    Loads a single CSV file into a pandas DataFrame.

    Parameters:
    - directory_path: The path to the directory containing the raw data files.
    - freq_number: The frequency number associated with the file.
    - file_type: The type of the node (client, node).
    - identifier: The identifier number for the specific node.

    Returns:
    - A DataFrame containing the data from the file
    and two additional columns with the frequency and the percentage of CPU utilization.
    """
    # Construct the file path and load the data
    file_path = f'{directory_path}/freq{freq_number}_{file_type}{identifier}_single_cpus_clean.csv'
    data = pd.read_csv(file_path, sep=' ', header=None)

    # Clean the data by removing the unix time column and setting appropriate column names
    data = data.drop(data.columns[0], axis=1)
    columns = ['time',
               'CPU',
               '%usr',
               '%nice',
               '%sys',
               '%iowait',
               '%irq',
               '%soft',
               '%steal',
               '%guest',
               '%gnice',
               '%idle'
               ]

    # Add extra columns: node type, frequency and percentage of CPU utilization
    data.columns = columns
    data[file_type] = f'{file_type}{identifier}'
    data['freq'] = freq_number
    data['%usage'] = 100 - data['%idle']

    return data


def preprocess_data(data, identifier):
    """
    Preprocess loaded data for analysis.

    Parameters:
    - data: DataFrame containing the raw data.
    - identifier: Column name to differentiate the data source (client, node).

    Returns:
    - A DataFrame containing the preprocessed data.
    """
    # Filter the data create two columns keeping the hour  the experiment took place
    # and the seconds each run was running for
    filtered_data = data[["time", "CPU", "%usage", identifier, "freq"]]
    filtered_data['time'] = pd.to_datetime(filtered_data['time'], format='%H:%M:%S').dt.time
    filtered_data['secs'] = np.floor(filtered_data.groupby([identifier, 'freq']).cumcount() / number_cpus) + 1

    return filtered_data


def load_data(frequencies, file_type):
    """
    Load and concatenate data for all specified frequencies and identifiers based on the file type.

    Parameters:
    - frequencies: List of frequency values to load data for.
    - file_type: Type of node (client, node).
    - directory_path: Base directory where data files are stored.

    Returns:
    - A DataFrame containing all the loaded and concatenated data.
    """
    # Determine the range of IDs based on node type
    if file_type == 'client':
        id_range = range(client_number)
    else:
        id_range = range(node_number)

    # Load and concatenate all the data files for the given frequencies and IDs
    dfs = [load_single_file(data_directory, freq, file_type, i) for freq in frequencies for i in id_range]

    return pd.concat(dfs, ignore_index=True)


def process_node_data(frequencies, node_type, start_sec, end_sec):
    """
    Load, preprocess, and filter data for a specific node type within a given time frame.

    Parameters:
    - frequencies: List of frequencies to include.
    - node_type: The type of the node ('client', 'node').
    - start_sec: Start second for the time frame filter.
    - end_sec: End second for the time frame filter.

    Returns:
    - Filtered and preprocessed DataFrame for the specified node type and time frame.
    """
    # Load and preprocess the data, then filter it based on the specified time frame
    data = load_data(frequencies, node_type)
    filtered_data = preprocess_data(data, node_type)
    filtered_data_timeframe = filtered_data[(filtered_data['secs'] >= start_sec) & (filtered_data['secs'] <= end_sec)]

    return filtered_data_timeframe


node_data_cpu_15s = process_node_data(frequencies, 'node', 13, 27)
node_data_cpu_15s_max = node_data_cpu_15s.groupby(['time', 'node', 'freq'])['%usage'].max().reset_index()


def load_single_file_network(directory_path, freq_number, file_type, identifier):
    """
    Loads a single CSV file into a pandas DataFrame.

    Parameters:
    - directory_path: The path to the directory containing the raw data files.
    - freq_number: The frequency number associated with the file.
    - file_type: The type of the node (client, node).
    - identifier: The identifier number for the specific node.

    Returns:
    - A DataFrame containing the data from the file
    """
    # Construct the file path and load the data
    file_path = f'{directory_path}/freq{freq_number}_{file_type}{identifier}_network.csv'
    data = pd.read_csv(file_path, sep=' ', skiprows=2).iloc[:, :3]
    data.columns = ['time'] + ['KB/s_in'] + ['KB/s_out']

    # change time form Unix
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data['time'] = data['time'].dt.time

    # Add extra columns: node type and frequency
    data[file_type] = f'{file_type}{identifier}'
    data['freq'] = freq_number

    return data

def process_network_data(data, network_type, start_seconds, end_seconds):
    """
    Process raw network data within a given time frame.

    Parameters:
    - data: A pandas DataFrame containing the raw network data.
    - network_type: The type of node (client, node).
    - start_sec: Start second for the time frame filter.
    - end_sec: End second for the time frame filter.

    Returns:
    - A pandas DataFrame containing the processed data for network throughput analysis.
    """
    data['secs'] = np.floor(data.groupby([network_type, 'freq']).cumcount()) + 1

    # Filter data based on the specified time range
    data_15s = data[(data['secs'] >= start_seconds) & (data['secs'] <= end_seconds)]
    data_15s['network_KB/s'] = data_15s['KB/s_in'] + data_15s['KB/s_out']

    # Changing KB/s to Mbps
    data_15s['network_Mbps'] = data_15s['network_KB/s'] / 125
    data_15s['Mbps_in'] = data_15s['KB/s_in'] / 125
    data_15s['Mbps_out'] = data_15s['KB/s_out'] / 125

    data_15s = data_15s[['time', 'network_Mbps', 'Mbps_in', 'Mbps_out', network_type, 'freq', 'secs']]

    return data_15s


def load_data_for_frequencies_network(frequencies, file_type, directory_path):
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

        # Determine the range of IDs based on node type
        if file_type == 'client':
            id_range = range(client_number)
        else:
            id_range = range(node_number)

        for i in id_range:
            dfs_for_current_freq.append(load_single_file_network(directory_path, freq, file_type, i))
        all_dfs.extend(dfs_for_current_freq)

    return pd.concat(all_dfs, ignore_index=True)


node_data_network = load_data_for_frequencies_network(frequencies, 'node', data_directory)
node_data_network_15s = process_network_data(node_data_network, 'node', 13, 27)
node_data_network_15s = node_data_network_15s[['time', 'network_Mbps', 'node', 'freq', 'secs']]


def plot_response_cpu_dot_plot(txs_received, node_cpu_data, output_path):
    """
     Generate and save a scatter plot comparing the response rate against max CPU utilisation for different
     request rates.

     Parameters:
     - txs_received: Filtered DataFrame containing transaction responses.
     - node_cpu_data: Filtered DataFrame containing CPU utilisation data of nodes.
     - output_path: The path where the output plot will be saved.
     """
    resp_cpu = txs_received.merge(node_cpu_data, left_on='time', right_on='time').drop('freq_y',axis=1)
    resp_cpu.columns = ['time', 'secs', 'freq', 'fresp', 'node', '%usage']

    # Create a custom color palette
    custom_palette = sns.color_palette("viridis", n_colors=len(resp_cpu['freq'].unique()))[::-1]
    sns.set_palette(custom_palette)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=resp_cpu, x='%usage', y='fresp', hue='freq', palette=custom_palette, s=50, alpha=0.6)

    plt.xlabel('Max CPU (%)', size=24)
    plt.ylabel('f$_{resp}$', size=26)
    plt.legend().remove()

    plt.xticks(size=22)
    plt.yticks(size=22)

    # Hide every second tick
    tick_labels = plt.xticks()[1]
    for i, label in enumerate(tick_labels):
        if i % 2 == 1:
            label.set_visible(False)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_plot_response_cpu_dot_plot.pdf'))

    plt.show()


plot_response_cpu_dot_plot(txs_received_per_second, node_data_cpu_15s_max, figures_directory)

def plot_response_network_dot_plot(txs_received, node_network_data, output_path):
    """
     Generate and save a scatter plot comparing the response rate against network utilisation for different
     request rates.

     Parameters:
     - txs_received: Filtered DataFrame containing transaction responses.
     - node_network_data: Filtered DataFrame containing network utilisation data of nodes.
     - output_path: The path where the output plot will be saved.
     """
    resp_network = txs_received.merge(node_network_data, left_on='time', right_on='time').drop(['freq_y', 'secs_y'],axis=1)
    resp_network.columns = ['time', 'secs', 'freq', 'fresp', 'network_Mbps', 'node']

    # Create a custom color palette
    custom_palette = sns.color_palette("viridis", n_colors=len(resp_network['freq'].unique()))[::-1]
    sns.set_palette(custom_palette)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=resp_network, x='network_Mbps', y='fresp', hue='freq', palette=custom_palette, s=50, alpha=0.6)

    plt.xlabel('Network (Mbps)', size=24)
    plt.ylabel('')
    plt.legend().remove()

    plt.xticks(size=22)
    plt.yticks(size=22)

    # Remove y-axis tick labels but keep the tick lines
    ax = plt.gca()
    ax.set_yticklabels([])

    # Hide every second tick
    tick_labels = plt.xticks()[1]
    for i, label in enumerate(tick_labels):
        if i % 2 == 1:
            label.set_visible(False)

    # Create a color bar with the mapping from colors to request rates
    original_palette = sns.color_palette("viridis", as_cmap=True)
    cmap = original_palette.reversed()
    norm = mpl.colors.Normalize(vmin=0, vmax=resp_network['freq'].max())
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ticks = np.arange(200, resp_network['freq'].max()+1, 400), spacing='proportional')
    cb.set_label('f$_{req}$', size=26)
    cb.ax.tick_params(labelsize=22)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_plot_response_network_dot_plot.pdf'))

    plt.show()


plot_response_network_dot_plot(txs_received_per_second, node_data_network_15s, figures_directory)





