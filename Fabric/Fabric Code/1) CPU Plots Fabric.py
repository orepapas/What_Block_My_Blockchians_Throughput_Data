import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import os

# Global constants for directory paths to organize data and figures
data_directory = 'What Blocks My Blockchain’s Throughput - Data/Fabric/DLPS - Fabric Raw Data/'
figures_directory = 'What Blocks My Blockchain’s Throughput - Data/Fabric/Figures/CPU/'

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


def load_single_file(directory_path, freq_number, file_type, identifier):
    """
    Loads a single CSV file into a pandas DataFrame.

    Parameters:
    - directory_path: The path to the directory containing the raw data files.
    - freq_number: The frequency number associated with the file.
    - file_type: The type of the node (client, peer, orderer).
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
    - identifier: Column name to differentiate the data source (client, peer, orderer).

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
    - file_type: Type of node (client, peer, orderer).
    - directory_path: Base directory where data files are stored.

    Returns:
    - A DataFrame containing all the loaded and concatenated data.
    """
    # Determine the range of IDs based on node type
    if file_type == 'client':
        id_range = range(client_number)
    elif file_type == 'peer':
        id_range = range(peer_number)
    else:
        id_range = range(orderer_number)

    # Load and concatenate all the data files for the given frequencies and IDs
    dfs = [load_single_file(data_directory, freq, file_type, i) for freq in frequencies for i in id_range]

    return pd.concat(dfs, ignore_index=True)


def process_node_data(frequencies, node_type, start_sec, end_sec):
    """
    Load, and filter data for a specific node type within a given time frame.

    Parameters:
    - frequencies: List of frequencies to include.
    - node_type: The type of the node ('client', 'peer', 'orderer').
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


# Process and filter the data for each node type for the given time frame
client_filtered_data_15s = process_node_data(frequencies, 'client', start_seconds, end_seconds)
peer_filtered_data_15s = process_node_data(frequencies, 'peer', start_seconds, end_seconds)
orderer_filtered_data_15s = process_node_data(frequencies, 'orderer', start_seconds, end_seconds)


def plot_cpu_dot_plot(client_data_15s, peer_data_15s, orderer_data_15s, output_path):
    """
    Generate and save a scatter plot comparing CPU usage across different frequencies and node types.

    Parameters:
    - client_data: DataFrame containing data for client nodes.
    - peer_data: DataFrame containing data for peer nodes.
    - orderer_data: DataFrame containing data for orderer nodes.
    - output_path: The path where the output plot will be saved.
    """
    # Prepare the data by calculating maximum CPU usage for each node and frequency per second
    # and combine all data into a single DataFrame for plotting
    max_usage_client = client_data_15s.groupby(['freq', 'client', 'secs'])['%usage'].max().reset_index()
    max_usage_peer = peer_data_15s.groupby(['freq', 'peer', 'secs'])['%usage'].max().reset_index()
    max_usage_orderer = orderer_data_15s.groupby(['freq', 'orderer', 'secs'])['%usage'].max().reset_index()

    max_usage_client['type'] = 'Client'
    max_usage_peer['type'] = 'Peer'
    max_usage_orderer['type'] = 'Orderer'

    max_usage_peer.rename(columns={'Peer': 'Client'}, inplace=True)
    max_usage_orderer.rename(columns={'Orderer': 'Client'}, inplace=True)

    combined_data = pd.concat([max_usage_client, max_usage_peer, max_usage_orderer], ignore_index=True)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=combined_data, x='freq', y='%usage', hue='type', palette=['#bb6894', '#92aed0', '#29ad99'],
                    s=50, alpha=0.6)

    plt.xlabel('')
    plt.ylabel('CPU (max core % usage per node)', size=24)
    plt.legend(title='', fontsize=24)

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
    plt.savefig(os.path.join(output_path, 'cpu_dot_plot.pdf'))

    plt.show()

    return combined_data


combined_data_dot_plot = plot_cpu_dot_plot(client_filtered_data_15s, peer_filtered_data_15s, orderer_filtered_data_15s,
                                           figures_directory)


def cpu_all_cores_plot(client_data_15s, peer_data_15s, orderer_data_15s, output_path):
    """
    Generate and save a boxplot comparing CPU usage across all cores for different node types and frequencies.

    Parameters:
    - client_data_15s: Filtered DataFrame containing data for clients in a 15 sec window.
    - peer_data_15s: Filtered DataFrame containing data for peers in a 15 sec window .
    - orderer_data_15s: Filtered DataFrame containing data for orderers in a 15 sec window .
    - output_path: The directory path where the output plot will be saved.
    """
    # Pre-process and combine data from different node types into a single DataFrame
    client_data_15s = client_data_15s[['freq', '%usage']]
    client_data_15s['instance'] = 'client'
    peer_data_15s = peer_data_15s[['freq', '%usage']]
    peer_data_15s['instance'] = 'peer'
    orderer_data_15s = orderer_data_15s[['freq', '%usage']]
    orderer_data_15s['instance'] = 'orderer'

    combined_data = pd.concat([client_data_15s, peer_data_15s, orderer_data_15s])

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    boxplot = sns.boxplot(x='freq', y='%usage', hue='instance', data=combined_data, palette=['#bb6894', '#92aed0', '#29ad99'])

    handles, _ = boxplot.get_legend_handles_labels()
    plt.legend(handles, ['Client', 'Peer', 'Orderer'], title='', fontsize="20")

    plt.xticks(size=20)
    plt.yticks(size=20)

    plt.xlabel('$f_{req}$', size=22)
    plt.ylabel('CPU (%)', size=22)
    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'cpu_all_cores_plot.pdf'))

    plt.show()

    return combined_data


combined_data_box_plot = cpu_all_cores_plot(client_filtered_data_15s, peer_filtered_data_15s, orderer_filtered_data_15s,
                                            figures_directory)


def mean_util_peer_plot(peer_data_15s, output_path):
    """
    Generates and saves a line plot showing the average CPU usage for each peer across different frequencies.

    Parameters:
    - peer_data_15s: Filtered DataFrame containing data for peers in a 15 sec window.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate and pivot the data to get the average CPU usage for each peer and frequency
    peer_data_15s = peer_data_15s[['freq', 'peer', '%usage']]
    peer_data_15s['peer_id'] = peer_data_15s['peer'].str.extract('(\d+)').astype(int)
    mean_node_util = peer_data_15s.groupby(['freq', 'peer_id'])['%usage'].mean().reset_index()
    pivoted_df = mean_node_util.pivot(index='freq', columns='peer_id', values='%usage')

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
    plt.ylabel('CPU (%)', size=24)
    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, 'mean_util_peer_plot.pdf'))

    plt.show()


mean_util_peer_plot(peer_filtered_data_15s, figures_directory)


def mean_util_core_plot(peer_data_15s, peer, output_path):
    """
    Generates and saves a line plot showing the average CPU usage for each core of a specified peer across different frequencies.

    Parameters:
    - peer_data_15s: Filtered DataFrame containing data for a specific peer in a 15 sec window.
    - peer: The specific peer to plot data for.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate and pivot the data to get the average CPU usage for each core and frequency
    peer_data_15s = peer_data_15s[['freq', 'peer', 'CPU', '%usage']]
    peer_data_15s = peer_data_15s[peer_data_15s['peer'] == peer]
    mean_peer_util = peer_data_15s.groupby(['freq', 'CPU'])['%usage'].mean().reset_index()
    pivoted_df = mean_peer_util.pivot(index='freq', columns='CPU', values='%usage')

    # Create a custom color palette
    custom_palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=(len(pivoted_df.columns)))
    sns.set_palette(custom_palette)

    # Create the line plot
    plt.figure(figsize=(8, 6))

    for column in pivoted_df.columns:
        sns.lineplot(x=pivoted_df.index, y=pivoted_df[column], linewidth=3)

    plt.title('')

    plt.xticks(size=22)
    plt.yticks(size=22)

    # Hide every second tick
    tick_labels = plt.xticks()[1]
    for i, label in enumerate(tick_labels):
        if i % 2 == 1:
            label.set_visible(False)

    plt.xlabel('$f_{req}$', size=26)
    plt.ylabel('CPU (%)', size=24)


    # Create a color bar with the mapping from colors to core IDs
    cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(pivoted_df.columns) - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ticks=[i for i in range(0, len(pivoted_df.columns), 2)], spacing='proportional')
    cb.set_label('Core id', size=24)
    cb.ax.tick_params(labelsize=22)

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, f'mean_util_core_plot_{peer}.pdf'))

    plt.show()


mean_util_core_plot(peer_filtered_data_15s, 'peer0', figures_directory)


def util_time_plot(peer_data_15s, peer, freq, output_path):
    """
    Generates and saves a line plot showing the CPU usage for each core of a specified peer and frequency over time.

    Parameters:
    - peer_data_15s: Filtered DataFrame containing data for a specific peer in a 15 sec window.
    - peer: The specific peer to plot data for.
    - freq: The specific frequency to plot data for.
    - output_path: The directory path where the output plot will be saved.
    """
    # Aggregate the data to get the average CPU usage for each core per second and calculate
    # mean CPU usage for each core
    peer_data_15s = peer_data_15s[peer_data_15s['freq'] == freq]
    peer_data_15s = peer_data_15s[peer_data_15s['peer'] == peer]
    peer_data_15s['secs'] = peer_data_15s['secs'] - 13
    peer_data_15s = peer_data_15s[['secs', 'CPU', '%usage']]
    mean_usage = peer_data_15s.groupby('CPU')['%usage'].mean().reset_index()

    # Create a custom color palette
    custom_palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=(len(peer_data_15s['CPU'].unique())))
    sns.set_palette(custom_palette)

    # Set up figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 15]})

    # Create the line plot
    for cpu in sorted(peer_data_15s['CPU'].unique()):
        cpu_data = peer_data_15s[peer_data_15s['CPU'] == cpu]
        sns.lineplot(x='secs', y='%usage', data=cpu_data, ax=ax2, linewidth=3)
    ax2.set_xlabel('Time (sec)', fontsize=22)
    ax2.tick_params(axis='both', labelsize=20)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    ax2.set(ylabel=None)
    ax2.set(yticklabels=[])

    # # Sync y-axis labels between subplots
    ax1.set_ylim(ax2.get_ylim())

    # Create a color bar with the mapping from colors to core IDs
    cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(peer_data_15s['CPU'].unique()) - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ticks=[i for i in range(0, len(peer_data_15s['CPU'].unique()), 2)], ax=ax2, spacing='proportional')
    cb.set_label('Core id', size=22)
    cb.ax.tick_params(labelsize=20)

    # Draw lines for average CPU usage
    for i, row in mean_usage.iterrows():
        ax1.axhline(y=row['%usage'], color=custom_palette[i], linewidth=3)

    ax1.set_ylabel('CPU (%)', fontsize=22)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_title('Mean', fontsize=15)
    ax1.set(xticklabels=[])

    plt.tight_layout()

    # Save the plot
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(os.path.join(output_path, f'util_time_plot_{peer}_{freq}.pdf'))

    plt.show()


util_time_plot(peer_filtered_data_15s, 'peer0', 1600, figures_directory)


