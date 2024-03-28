import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os

# Global constants for directory paths to organize data and figures
data_directory = 'What Blocks My Blockchain’s Throughput - Data/Fabric/DLPS - Fabric Raw Data/'
figures_directory = 'What Blocks My Blockchain’s Throughput - Data/Fabric/Figures/Network/'

# General settings for the analysis
client_number = 16
peer_number = 8
orderer_number = 4
instance_memory = 67108864
number_cpus = 16
frequencies = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]


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
    - file_type: The type of the node (client, peer, orderer).
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
    - network_type: The type of node (client, peer, orderer).
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
    - file_type: Type of node (client, peer, orderer).
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
        elif file_type == 'peer':
            id_range = range(peer_number)
        else:
            id_range = range(orderer_number)

        for i in id_range:
            dfs_for_current_freq.append(load_single_file_network(directory_path, freq, file_type, i))
        all_dfs.extend(dfs_for_current_freq)

    return pd.concat(all_dfs, ignore_index=True)


client_data_network = load_data_for_frequencies_network(frequencies, 'client', data_directory)
peer_data_network = load_data_for_frequencies_network(frequencies, 'peer', data_directory)
orderer_data_network = load_data_for_frequencies_network(frequencies, 'orderer',data_directory)

client_data_network_15s = process_network_data(client_data_network, 'client', start_seconds, end_seconds)
peer_data_network_15s = process_network_data(peer_data_network, 'peer', start_seconds, end_seconds)
orderer_data_network_15s = process_network_data(orderer_data_network, 'orderer', start_seconds, end_seconds)


def plot_network_dot_plot(client_data_15s, peer_data_15s, orderer_data_15s, output_path):
    """
    Generate and save a scatter plot comparing network utilization across different frequencies and node types.

    Parameters:
    - client_data: DataFrame containing data for client nodes.
    - peer_data: DataFrame containing data for peer nodes.
    - orderer_data: DataFrame containing data for orderer nodes.
    - output_path: The path where the output plot will be saved.
    """
    # Prepare the data by calculating mean network usage for each node and frequency per second
    # and combine all data into a single DataFrame for plotting
    network_usage_client = client_data_15s.groupby(['freq', 'client', 'secs'])['network_Mbps'].mean().reset_index()
    network_usage_peer = peer_data_15s.groupby(['freq', 'peer', 'secs'])['network_Mbps'].mean().reset_index()
    network_usage_orderer = orderer_data_15s.groupby(['freq', 'orderer', 'secs'])['network_Mbps'].mean().reset_index()

    network_usage_client['type'] = 'Client'
    network_usage_peer['type'] = 'Peer'
    network_usage_orderer['type'] = 'Orderer'

    network_usage_peer.rename(columns={'Peer': 'Client'}, inplace=True)
    network_usage_orderer.rename(columns={'Orderer': 'client'}, inplace=True)

    combined_data = pd.concat([network_usage_peer, network_usage_orderer, network_usage_client], ignore_index=True)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=combined_data, x='freq', y='network_Mbps', hue='type', palette=['#92aed0', '#29ad99', '#bb6894'],
                    s=50, alpha=0.6)

    plt.xlabel('')
    plt.ylabel('Network (Mbps)', size=24)
    plt.legend().remove()

    x_ticks_locations = plt.xticks()[0][1:-1]
    plt.xticks(x_ticks_locations, [''] * len(x_ticks_locations))
    plt.yticks(size=22)

    # Hide every second tick
    tick_labels = plt.xticks()[1]
    for i, label in enumerate(tick_labels):
        if i % 2 == 1:
            label.set_visible(False)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'network_dot_plot.pdf'))

    plt.show()

    return combined_data


combined_network_dot_plot = plot_network_dot_plot(client_data_network_15s, peer_data_network_15s,
                                                  orderer_data_network_15s, figures_directory)


def plot_network_traffic_orderer(data_15s, output_path):
    """
    Generates and saves a line plot showing inbound/outbound network traffic for each orderer across different frequencies

    Parameters:
    - data_15s: Filtered DataFrame containing network data of orderers.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate and pivot the data to get the average inbound/outbound network traffic for each orderer and frequency
    data_15s = data_15s[['freq', 'orderer', 'Mbps_in', 'Mbps_out']]
    data_15s['orderer_id'] = data_15s['orderer'].str.extract('(\d+)').astype(int)
    mean_traffic_in = data_15s.groupby(['freq', 'orderer_id'])['Mbps_in'].mean().reset_index()
    mean_traffic_out = data_15s.groupby(['freq', 'orderer_id'])['Mbps_out'].mean().reset_index()
    pivoted_in = mean_traffic_in.pivot(index='freq', columns='orderer_id', values='Mbps_in')
    pivoted_out = mean_traffic_out.pivot(index='freq', columns='orderer_id', values='Mbps_out')

    # Create a custom color palette
    custom_palette = sns.color_palette("dark:#5A9_r", as_cmap=False, n_colors=(len(pivoted_in.columns)))
    sns.set_palette(custom_palette)


    plt.figure(figsize=(8, 6))

    # Create the line plot
    for i, column in enumerate(pivoted_in.columns):
        color = custom_palette[i]
        sns.lineplot(x=pivoted_in.index, y=pivoted_in[column], linestyle='--', linewidth=3, color=color )
        sns.lineplot(x=pivoted_out.index, y=pivoted_out[column], label=f'{column}', linewidth=3, color=color)

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('Network (Mbps)', size=24)
    plt.legend(title='Orderer id', fontsize=20, title_fontsize=20)

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
    plt.savefig(os.path.join(output_path, 'inbound_outbound_orderer_plot.pdf'))

    plt.show()


plot_network_traffic_orderer(orderer_data_network_15s, figures_directory)


def orderer_ratio_network_plot(orderer_data_15s, output_path):
    """
    Generates and saves a line plot showing the ratio between the outbound network traffic for each follower orderer
    and the leader across different frequencies

    Parameters:
    - orderer_data_15s: Filtered DataFrame containing network data of orderers.
    - output_path: The directory path where the output plot will be saved.
    """
    orderer_data_15s = orderer_data_15s[['freq', 'orderer', 'Mbps_in', 'Mbps_out']]
    orderer_data_15s['orderer_id'] = orderer_data_15s['orderer'].str.extract('(\d+)').astype(int)
    mean_orderer_traffic_out = orderer_data_15s.groupby(['freq', 'orderer_id'])['Mbps_out'].mean().reset_index()
    pivoted_orderer_out = mean_orderer_traffic_out.pivot(index='freq', columns='orderer_id', values='Mbps_out')
    pivoted_orderer_out = (1 / pivoted_orderer_out.div(pivoted_orderer_out.iloc[:, 0], axis=0)).iloc[:, 1:]

    # Create a custom color palette
    custom_palette = sns.color_palette("dark:#5A9_r", as_cmap=False, n_colors=(len(pivoted_orderer_out.columns)))
    sns.set_palette(custom_palette)

    plt.figure(figsize=(8, 6))

    for i, column in enumerate(pivoted_orderer_out.columns):
        plt.plot(pivoted_orderer_out.index, pivoted_orderer_out[column], label=f'{column}')

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('Outbound traffic ratio', size=24)
    plt.legend(title='Orderer id', fontsize=20, title_fontsize=20)

    # Set the y-axis ticks
    plt.xticks(size=22)
    plt.ylim(0, 4.5)
    plt.yticks([0, 1, 2, 3, 4], size=22)

    # Hide every second tick
    tick_labels = plt.xticks()[1]
    for i, label in enumerate(tick_labels):
        if i % 2 == 1:
            label.set_visible(False)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'ratio_network_plot.pdf'))

    plt.show()


orderer_ratio_network_plot(orderer_data_network_15s, figures_directory)


def plot_network_traffic_peer(data_15s, output_path):
    """
    Generates and saves a line plot showing inbound/outbound network traffic for each peer across different frequencies

    Parameters:
    - data_15s: Filtered DataFrame containing network data of peers.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate and pivot the data to get the average inbound/outbound network usage for each peer and frequency
    data_15s = data_15s[['freq', 'peer', 'Mbps_in', 'Mbps_out']]
    data_15s['peer_id'] = data_15s['peer'].str.extract('(\d+)').astype(int)
    mean_traffic_in = data_15s.groupby(['freq', 'peer_id'])['Mbps_in'].mean().reset_index()
    mean_traffic_out = data_15s.groupby(['freq', 'peer_id'])['Mbps_out'].mean().reset_index()
    pivoted_in = mean_traffic_in.pivot(index='freq', columns='peer_id', values='Mbps_in')
    pivoted_out = mean_traffic_out.pivot(index='freq', columns='peer_id', values='Mbps_out')

    colors = ['#6d77aa', '#de5c5b']

    plt.figure(figsize=(8, 6))

    # Create the line plot
    for i, column in enumerate(pivoted_in.columns):
        color_index = i % 2
        color = colors[color_index]
        sns.lineplot(x=pivoted_in.index, y=pivoted_in[column], linestyle='--', linewidth=3, color=color)
        sns.lineplot(x=pivoted_out.index, y=pivoted_out[column], label=f'{column}', linewidth=3, color=color)

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('Network (Mbps)', size=24)
    plt.legend(title='Peer id', fontsize=20, title_fontsize=20)

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
    plt.savefig(os.path.join(output_path, 'inbound_outbound_peer_plot.pdf'))

    plt.show()


plot_network_traffic_peer(peer_data_network_15s, figures_directory)


def plot_network_traffic_peer_no_block(peer_data_15s, orderer_data_15s, output_path):
    """
    Generates and saves a line plot showing outbound network traffic for each peer excluding the traffic generated
    by the block propagation across different frequencies

    Parameters:
    - peer_data_15s: Filtered DataFrame containing network data of peers.
    - orderer_data_15s: Filtered DataFrame containing network data of orderers.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate and pivot the data to get the average outbound network usage for each peer and frequency
    peer_data_15s = peer_data_15s[['freq', 'peer', 'Mbps_in', 'Mbps_out']]
    peer_data_15s['peer_id'] = peer_data_15s['peer'].str.extract('(\d+)').astype(int)
    mean_traffic_out = peer_data_15s.groupby(['freq', 'peer_id'])['Mbps_out'].mean().reset_index()
    pivoted_out = mean_traffic_out.pivot(index='freq', columns='peer_id', values='Mbps_out')

    orderer_data_15s = orderer_data_15s[['freq', 'orderer', 'Mbps_in', 'Mbps_out']]
    orderer_data_15s['orderer_id'] = orderer_data_15s['orderer'].str.extract('(\d+)').astype(int)
    orderer_data_15s = orderer_data_15s[orderer_data_15s['orderer_id'] != 0]
    mean_orderer_traffic_out = orderer_data_15s.groupby(['freq'])['Mbps_out'].mean().reset_index()

    for i in range(len(pivoted_out.columns)):
        if i % 2 == 0:
            pivoted_out.iloc[:, i] = pivoted_out.iloc[:, i].values - mean_orderer_traffic_out.iloc[:, 1].values

    colors = ['#6d77aa', '#de5c5b']

    plt.figure(figsize=(8, 6))

    # Create the line plot
    for i, column in enumerate(pivoted_out.columns):
        color_index = i % 2
        color = colors[color_index]
        sns.lineplot(x=pivoted_out.index, y=pivoted_out[column], linestyle='--', linewidth=3, color=color)
        sns.lineplot(x=pivoted_out.index, y=pivoted_out[column], label=f'{column}', linewidth=3, color=color)

    plt.xlabel('f$_{req}$', size=26)
    plt.ylabel('Network (Mbps)', size=24)
    plt.legend(title='Peer id', fontsize=20, title_fontsize=20)

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
    plt.savefig(os.path.join(output_path, 'plot_network_traffic_peer_no_block.pdf'))

    plt.show()


plot_network_traffic_peer_no_block(peer_data_network_15s, orderer_data_network_15s, figures_directory)


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
    tick_labels = plt.xticks()[1]
    for i, label in enumerate(tick_labels):
        if i % 2 == 1:
            label.set_visible(False)

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
    plt.savefig(os.path.join(output_path, 'inbound_outbound_client_plot.pdf'))

    plt.show()


plot_network_traffic_client(client_data_network_15s, figures_directory)



