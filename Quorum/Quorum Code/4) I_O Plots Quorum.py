import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os

# Global constants for directory paths to organize data and figures
data_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/DLPS - Quorum Raw Data'
figures_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/Figures/I_O/'

# General settings for the analysis
client_number = 16
node_number = 8
instance_memory = 67108864
number_cpus = 16
frequencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
               1900, 2000, 2100, 2200, 2300, 2400, 2500]

# Define the time frame for analysis
start_seconds = 13
end_seconds = 27


def ensure_dir(directory):
    """
    Ensure that the specified directory exists, creating it if necessary.

    Parameters:
    - directory: The path of the directory to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_single_file_io(directory_path, freq_number, file_type, identifier):
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
    file_path = f'{directory_path}/freq{freq_number}_{file_type}{identifier}_resources.csv'
    data = pd.read_csv(file_path, sep=' ', header=1)


    data = transform_data(data)

    # Add extra columns: node type and frequency
    data[file_type] = f'{file_type}{identifier}'
    data['freq'] = freq_number

    return data


def transform_data(data):
    """
    Transforms raw memory data frame into a structured format with multi-level columns.

    Parameters:
    - data: The raw I/O pandas DataFrame to be transformed.

    Returns:
    - A DataFrame with structured multi-level columns and formatted time.
    """
    # Rename the first column to 'time'
    first_column_name = data.columns[0]
    data.rename(columns={first_column_name: 'time'}, inplace=True)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data['time'] = data['time'].dt.time

    columns = pd.MultiIndex.from_tuples([
        ('time', 'time'),
        ('procs', 'r'),
        ('procs', 'b'),
        ('memory', 'swpd'),
        ('memory', 'free'),
        ('memory', 'inact'),
        ('memory', 'active'),
        ('swap', 'si'),
        ('swap', 'so'),
        ('io', 'bi'),
        ('io', 'bo'),
        ('system', 'in'),
        ('system', 'cs'),
        ('cpu', 'us'),
        ('cpu', 'sy'),
        ('cpu', 'id'),
        ('cpu', 'wa'),
        ('cpu', 'st')
    ], names=['Group', 'Subgroup'])

    data.columns = columns

    return data


def load_data_for_frequencies_io(frequencies, file_type, directory_path):
    """
    Load and concatenate data for all specified frequencies and node type.

    Parameters:
    - frequencies: List of frequency values to load data for.
    - file_type: Type of node (client, peer, orderer).
    - directory_path: Base directory where data files are stored.

    Returns:
    - A DataFrame containing all the data.
    """
    all_dfs = []
    for freq in frequencies:
        dfs_for_current_freq = []

        # Determine the range based on the file_type
        if file_type == 'client':
            id_range = range(client_number)
        else:
            id_range = range(node_number)

        for i in id_range:
            dfs_for_current_freq.append(load_single_file_io(directory_path, freq, file_type, i))
        all_dfs.extend(dfs_for_current_freq)

    return pd.concat(all_dfs, ignore_index=True)


client_data_resources = load_data_for_frequencies_io(frequencies, 'client', data_directory)
node_data_resources = load_data_for_frequencies_io(frequencies, 'node', data_directory)


def process_io_data(data, file_type, start_seconds, end_seconds):
    """
    Processes I/O usage data for a specific node type within a given time range.

    Parameters:
    - data: The DataFrame containing I/O usage data.
    - file_type: The type of the node (e.g., 'client', 'node').
    - instance_memory: Total memory of the instance.
    - start_seconds: Start second for the time window for analysis.
    - end_seconds: End second for the time window for analysis.

    Returns:
    - A DataFrame filtered by the specified time window and augmented with memory usage percentage.
    """
    data['secs'] = np.floor(data.groupby([file_type, 'freq']).cumcount())+1
    data_io = data[[('io', 'bi'), ('io', 'bo'), (file_type, ''), ('freq', ''), ('secs', '')]]
    data_io.columns = ['io_bytes_in', 'io_bytes_out', file_type, 'freq', 'secs']

    data_io_15s = data_io[(data_io['secs'] >= start_seconds) & (data_io['secs'] <= end_seconds)]

    # Calculate combined I/O and select relevant columns
    data_io_15s['io'] = (data_io_15s['io_bytes_in'] + data_io_15s['io_bytes_out']) / 10000
    data_io_15s = data_io_15s[['io', file_type, 'freq', 'secs']]

    return data_io_15s


client_data_io_15s = process_io_data(client_data_resources, 'client', start_seconds, end_seconds)
node_data_io_15s = process_io_data(node_data_resources, 'node', start_seconds, end_seconds)


def plot_io_dot_plot(client_data_15s, node_data_15s, output_path):
    """
     Generate and save a scatter plot comparing I/O utilization across different frequencies and node types.

     Parameters:
     - client_data_15s: Filtered DataFrame containing I/O data for a specific client in a 15 sec window.
     - node_data_15s: Filtered DataFrame containing I/O data for a specific node in a 15 sec window.
     - output_path: The path where the output plot will be saved.
     """
    # Prepare the data by calculating mean I/O usage for each node and frequency per second
    # and combine all data into a single DataFrame for plotting
    io_usage_client = client_data_15s.groupby(['freq', 'client', 'secs'])['io'].mean().reset_index()
    io_usage_node = node_data_15s.groupby(['freq', 'node', 'secs'])['io'].mean().reset_index()

    io_usage_client['type'] = 'client'
    io_usage_node['type'] = 'node'

    io_usage_node.rename(columns={'node': 'client'}, inplace=True)

    # Concatenate the two DataFrames
    combined_data = pd.concat([io_usage_client, io_usage_node], ignore_index=True)

    # Now let's plot the data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=combined_data, x='freq', y='io', hue='type', palette=['#bb6894', '#92aed0'],
                    s=50, alpha=0.6, )

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('I/O (%)', size=24)
    plt.legend().remove()

    plt.xticks(size=22)
    plt.yticks(size=22)

    # # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_i_o_dot_plot.pdf'))

    plt.show()

    return combined_data


combined_io_data = plot_io_dot_plot(client_data_io_15s, node_data_io_15s, figures_directory)


def plot_mean_io_node(node_data_15s, output_path):
    """
    Generate and save a line plot comparing average I/O utilization across different frequencies for peers.

    Parameters:
    - peer_data_15s: Filtered DataFrame containing memory data for peers in a 15 sec window.
    - output_path: The path where the output plot will be saved.
    """
    node_data_15s = node_data_15s[['freq', 'node', 'io']]
    node_data_15s['node_id'] = node_data_15s['node'].str.extract('(\d+)').astype(int)
    mean_node_io = node_data_15s.groupby(['freq', 'node_id'])['io'].mean().reset_index()
    pivoted_df_node = mean_node_io.pivot(index='freq', columns='node_id', values='io')

    plt.figure(figsize=(8, 6))
    custom_palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=(len(pivoted_df_node.columns)))
    sns.set_palette(custom_palette)
    for column in pivoted_df_node.columns:
        plt.plot(pivoted_df_node.index, pivoted_df_node[column], label=f'{column}', linewidth=3)

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('I/O (%)', size=24)
    plt.legend(title='Node id', fontsize=20, title_fontsize=20)
    plt.xticks(size=22)
    plt.yticks(size=22)

    # # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_node_mean_io_plot.pdf'))
    plt.show()


plot_mean_io_node(node_data_io_15s, figures_directory)

