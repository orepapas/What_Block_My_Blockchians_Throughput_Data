import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os

# Global constants for directory paths to organize data and figures
data_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/DLPS - Quorum Raw Data'
figures_directory = 'What Blocks My Blockchain’s Throughput - Data/Quorum/Figures/Network/'

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

    data_15s = data_15s[['network_Mbps', 'Mbps_in', 'Mbps_out', network_type, 'freq', 'secs']]

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

        # Determine the range based on the file_type
        if file_type == 'client':
            id_range = range(client_number)
        else:
            id_range = range(node_number)

        for i in id_range:
            dfs_for_current_freq.append(load_single_file_network(directory_path, freq, file_type, i))
        all_dfs.extend(dfs_for_current_freq)

    return pd.concat(all_dfs, ignore_index=True)


client_data_network = load_data_for_frequencies_network(frequencies, 'client', data_directory)
node_data_network = load_data_for_frequencies_network(frequencies, 'node', data_directory)

client_data_network_15s = process_network_data(client_data_network, 'client', start_seconds, end_seconds)
node_data_network_15s = process_network_data(node_data_network, 'node', start_seconds, end_seconds)


def plot_network_dot_plot(client_data_15s, node_data_15s, output_path):
    """
    Generate and save a scatter plot comparing network utilization across different frequencies and node types.

    Parameters:
    - client_data_15s: DataFrame containing data for client nodes.
    - node_data_15s: DataFrame containing data for peer nodes.
    - output_path: The path where the output plot will be saved.
    """
    # Prepare the data by calculating mean network usage for each node and frequency per second
    # and combine all data into a single DataFrame for plotting
    network_usage_client = client_data_15s.groupby(['freq', 'client', 'secs'])['network_Mbps'].mean().reset_index()
    network_usage_node = node_data_15s.groupby(['freq', 'node', 'secs'])['network_Mbps'].mean().reset_index()

    network_usage_client['type'] = 'Client'
    network_usage_node['type'] = 'Node'

    network_usage_node.rename(columns={'node': 'Client'}, inplace=True)

    combined_data = pd.concat([network_usage_node, network_usage_client], ignore_index=True)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=combined_data, x='freq', y='network_Mbps', hue='type', palette=['#92aed0', '#bb6894'],
                    s=50, alpha=0.6)

    plt.xlabel('')
    plt.ylabel('Network (Mbps)', size=24)
    plt.legend().remove()

    x_ticks_locations = plt.xticks()[0][1:-1]
    plt.xticks(x_ticks_locations, [''] * len(x_ticks_locations))
    plt.yticks(size=22)

    # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_network_dot_plot.pdf'))

    plt.show()

    return combined_data


combined_network_dot_plot = plot_network_dot_plot(client_data_network_15s, node_data_network_15s, figures_directory)


def plot_network_traffic_node(data_15s, output_path):
    """
    Generates and saves a line plot showing inbound/outbound network traffic for each node across different frequencies

    Parameters:
    - data_15s: Filtered DataFrame containing network data of nodes.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate and pivot the data to get the average inbound/outbound network traffic for each node and frequency
    data_15s = data_15s[['freq', 'node', 'Mbps_in', 'Mbps_out']]
    data_15s['node_id'] = data_15s['node'].str.extract('(\d+)').astype(int)
    mean_traffic_in = data_15s.groupby(['freq', 'node_id'])['Mbps_in'].mean().reset_index()
    mean_traffic_out = data_15s.groupby(['freq', 'node_id'])['Mbps_out'].mean().reset_index()
    pivoted_in = mean_traffic_in.pivot(index='freq', columns='node_id', values='Mbps_in')
    pivoted_out = mean_traffic_out.pivot(index='freq', columns='node_id', values='Mbps_out')

    # Create a custom color palette
    custom_palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=(len(pivoted_in.columns)))
    sns.set_palette(custom_palette)


    colors = ['#00b300', '#6d77aa', '#de5c5b']

    # Create the line plot
    plt.figure(figsize=(8, 6))

    # Loop through each column with an index
    for index, column in enumerate(pivoted_out.columns):
        # Select the color based on the column index
        if index == 0:
            color = colors[0]
        elif 1 <= index <= 3:
            color = colors[1]
        else:
            color = colors[2]

        sns.lineplot(x=pivoted_in.index, y=pivoted_in[column], linestyle='--', linewidth=3, color=color )
        sns.lineplot(x=pivoted_out.index, y=pivoted_out[column], label=f'{column}', linewidth=3, color=color)

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('Network (Mbps)', size=24)
    plt.legend(title='Node id', fontsize=20, title_fontsize=20)

    plt.xticks(size=22)
    plt.yticks(size=22)

    # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_inbound_outbound_node_plot.pdf'))

    plt.show()


plot_network_traffic_node(node_data_network_15s, figures_directory)


def plot_network_traffic_client(data_15s, output_path):
    """
    Generates and saves a line plot showing inbound/outbound network traffic for each client across different frequencies

    Parameters:
    - data_15s: Filtered DataFrame containing network data of clients.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate and pivot the data to get the average inbound/outbound network usage for each client and frequency
    data_15s = data_15s[['freq', 'client', 'Mbps_in', 'Mbps_out']]
    data_15s['client_id'] = data_15s['client'].str.extract('(\d+)').astype(int)
    mean_traffic_in = data_15s.groupby(['freq', 'client_id'])['Mbps_in'].mean().reset_index()
    mean_traffic_out = data_15s.groupby(['freq', 'client_id'])['Mbps_out'].mean().reset_index()
    pivoted_in = mean_traffic_in.pivot(index='freq', columns='client_id', values='Mbps_in')
    pivoted_out = mean_traffic_out.pivot(index='freq', columns='client_id', values='Mbps_out')

    # Create a custom color palette
    custom_palette = sns.color_palette("ch:", as_cmap=False, n_colors=(len(pivoted_in.columns)))
    sns.set_palette(custom_palette)

    plt.figure(figsize=(10, 6))

    # Create the line plot
    for i, column in enumerate(pivoted_in.columns):
        color = custom_palette[i]
        sns.lineplot(x=pivoted_in.index, y=pivoted_in[column], linestyle='--', linewidth=3, color=color)
        sns.lineplot(x=pivoted_out.index, y=pivoted_out[column], linewidth=3, color=color)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    plt.xlabel('f$_{req}$', size=22)
    plt.ylabel('Network (Mbps)', size=22)

    plt.xticks(size=20)
    plt.yticks(size=20)

    # Hide every second tick
    # tick_labels = plt.xticks()[1]
    # for i, label in enumerate(tick_labels):
    #     if i % 2 == 1:
    #         label.set_visible(False)

    # Create a color bar with the mapping from colors to client IDs
    cmap = sns.color_palette("ch:", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(pivoted_in.columns) - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ticks=[i for i in range(0, len(pivoted_in.columns), 2)], spacing='proportional')
    cb.set_label('Client id', size=22)
    cb.ax.tick_params(labelsize=20)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'q_inbound_outbound_client_plot.pdf'))

    plt.show()


plot_network_traffic_client(client_data_network_15s, figures_directory)



