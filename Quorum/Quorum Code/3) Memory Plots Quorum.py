import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os

# Global constants for directory paths to organize data and figures
data_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/DLPS - Quorum Raw Data'
figures_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/Figures/Memory/'

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


def load_single_file_memory(directory_path, freq_number, file_type, identifier):
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
    file_path = f'{directory_path}/freq{freq_number}_{file_type}{identifier}_resources.csv'
    data = pd.read_csv(file_path, sep=' ', header=1)

    # Prepare data transformations
    data = transform_data(data)

    # Add extra columns: node type and frequency
    data[file_type] = f'{file_type}{identifier}'
    data['freq'] = freq_number

    return data


def transform_data(data):
    """
    Transforms raw memory data frame into a structured format with multi-level columns.

    Parameters:
    - data: The raw memory pandas DataFrame to be transformed.

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


def load_data_for_frequencies_memory(frequencies, file_type, directory_path):
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

        # Determine the range based on the file_type
        if file_type == 'client':
            id_range = range(client_number)
        else:
            id_range = range(node_number)

        for i in id_range:
            dfs_for_current_freq.append(load_single_file_memory(directory_path, freq, file_type, i))
        all_dfs.extend(dfs_for_current_freq)

    return pd.concat(all_dfs, ignore_index=True)


client_data_resources = load_data_for_frequencies_memory(frequencies, 'client', data_directory)
node_data_resources = load_data_for_frequencies_memory(frequencies, 'node', data_directory)


def process_memory_data(data, file_type, instance_memory, start_seconds, end_seconds):
    """
    Processes memory usage data for a specific node type within a given time range.

    Parameters:
    - data: The DataFrame containing memory usage data.
    - file_type: The type of the node (e.g., 'client', 'node').
    - instance_memory: Total memory of the instance.
    - start_seconds: Start second for the time window for analysis.
    - end_seconds: End second for the time window for analysis.

    Returns:
    - A DataFrame filtered by the specified time window and augmented with memory usage percentage.
    """
    memory_data = data[[('time', 'time'), ('memory', 'free'), (file_type, ''), ('freq', '')]]
    memory_data.columns = ['time', 'free_memory', file_type, 'freq']
    memory_data['memory_used_%'] = (instance_memory - memory_data['free_memory'])*100/instance_memory
    memory_data['secs'] = np.floor(memory_data.groupby([file_type, 'freq']).cumcount()) + 1
    memory_data = memory_data[(memory_data['secs'] >= start_seconds) & (memory_data['secs'] <= end_seconds)]

    return memory_data


client_data_memory_15s = process_memory_data(client_data_resources, 'client', instance_memory, start_seconds, end_seconds)
node_data_memory_15s = process_memory_data(node_data_resources, 'node', instance_memory, start_seconds, end_seconds)


def plot_memory_dot_plot(client_data_15s, node_data_15s, output_path):
    """
     Generate and save a scatter plot comparing memory utilization across different frequencies and node types.

     Parameters:
     - client_data_15s: Filtered DataFrame containing memory data for a specific client in a 15 sec window.
     - node_data_15s: Filtered DataFrame containing memory data for a specific node in a 15 sec window.
     - output_path: The path where the output plot will be saved.
     """
    # Prepare the data by calculating mean memory usage for each node and frequency per second
    # and combine all data into a single DataFrame for plotting
    memory_usage_client = client_data_15s.groupby(['freq', 'client', 'secs'])['memory_used_%'].mean().reset_index()
    memory_usage_node = node_data_15s.groupby(['freq', 'node', 'secs'])['memory_used_%'].mean().reset_index()

    memory_usage_client['type'] = 'client'
    memory_usage_node['type'] = 'node'

    # Renaming the columns to match for concatenation
    memory_usage_node.rename(columns={'node': 'client'}, inplace=True)

    # Concatenate the two DataFrames
    combined_data = pd.concat([memory_usage_node, memory_usage_client], ignore_index=True)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=combined_data, x='freq', y='memory_used_%', hue='type', palette=['#92aed0', '#bb6894'],
                    s=50, alpha=0.6)

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('Memory (%)', size=24)
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
    plt.savefig(os.path.join(output_path, 'q_memory_dot_plot.pdf'))

    plt.show()

    return combined_data


combined_memory_data = plot_memory_dot_plot(client_data_memory_15s, node_data_memory_15s, figures_directory)


def plot_peer_memory_usage(node_data_15s, output_path):
    """
    Generate and save a line plot for peer memory utilization.

    Parameters:
    - peer_data_15s: Filtered DataFrame containing memory data for peers in a 15 sec window.
    - output_path: The path where the output plot will be saved.
    """
    node_data_15s = node_data_15s[['freq', 'node', 'memory_used_%']]
    node_data_15s['node_id'] = node_data_15s['node'].str.extract('(\d+)').astype(int)
    mean_node_io = node_data_15s.groupby(['freq', 'node_id'])['memory_used_%'].mean().reset_index()
    pivoted_df_node = mean_node_io.pivot(index='freq', columns='node_id', values='memory_used_%')

    plt.figure(figsize=(8, 6))
    custom_palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=(len(pivoted_df_node.columns)))
    sns.set_palette(custom_palette)

    for column in pivoted_df_node.columns:
        plt.plot(pivoted_df_node.index, pivoted_df_node[column], label=f'{column}', linewidth=3)

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('Memory (%)', size=24)
    plt.legend(title='Node id', fontsize=20, title_fontsize=20, loc='upper left')
    plt.xticks(size=22)
    plt.yticks(size=22)

    # # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_plot_node_memory_usage.pdf'))
    plt.show()

plot_peer_memory_usage(node_data_memory_15s, figures_directory)

