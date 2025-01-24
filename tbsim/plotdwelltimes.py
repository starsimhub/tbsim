import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter



# TODO: Kaplan-Meier
def plot_kaplan_meier(data, dwell_time_col, event_observed_col=None):
    """
    Plots a Kaplan-Meier survival curve for the given data.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing survival data.
        dwell_time_col (str): Column name representing dwell times.
        event_observed_col (str, optional): Column indicating if the event was observed (1) or censored (0).
            If None, assumes all events are observed.

    Returns:
        None: Displays the Kaplan-Meier survival plot.
    """
    # Prepare the data
    durations = data[dwell_time_col]
    event_observed = data[event_observed_col] if event_observed_col else [1] * len(data)

    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # Fit the data
    kmf.fit(durations, event_observed=event_observed)

    # Plot the survival function
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("TBSim Kaplan-Meier Survival Curve", fontsize=16)
    plt.xlabel(f"Time ({dwell_time_col})", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.grid(True)
    plt.show()

# looks good
def state_transition_matrix(file_path=None, dwell_time_logger=None):
    """
    Generates and plots a state transition matrix from the provided data.
    Parameters:
    file_path (str, optional): Path to the CSV file containing the data. The CSV file should have columns 'agent_id' and 'state'.
    dwell_time_logger (pd.DataFrame, optional): DataFrame containing the data. Should have columns 'agent_id' and 'state'.
    Returns:
    None: The function plots the state transition matrix using seaborn's heatmap.
    Notes:
    - This is using plain 'state' columns recorded in the data - no need to 'going_to_state' column.
    - If both file_path and dwell_time_logger are provided, file_path will be used.
    - If neither file_path nor dwell_time_logger are provided, the function will print "No data provided." and return.
    - The transition matrix is normalized to show proportions. To display raw counts, comment out the normalization step.
    """

    if file_path is not None:
        df = pd.read_csv(file_path, na_values=[], keep_default_na=False)
    elif dwell_time_logger is not None:
        df = dwell_time_logger
    else:
        print("No data provided.")
        return

    # Create a transition matrix
    # Get the unique states
    unique_states = sorted(df['state_name'].dropna().unique())

    # Initialize a matrix of zeros
    transition_matrix = pd.DataFrame(
        data=0, index=unique_states, columns=unique_states, dtype=int
    )

    # Fill the matrix with transitions
    for agent_id, group in df.groupby('agent_id'):
        states = group['state_name'].values
        for i in range(len(states) - 1):
            transition_matrix.loc[states[i], states[i + 1]] += 1

    # Normalize rows to show proportions (optional, can comment this out if counts are preferred)
    transition_matrix_normalized = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

    # Plot the state transition matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        transition_matrix_normalized, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=unique_states, 
        yticklabels=unique_states
    )
    plt.title("State Transition Matrix (Normalized)")
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.show()

#looks better- but still not perfect
def sankey(file_path=None, dwell_time_logger=None):
    import plotly.graph_objects as go

    if file_path is not None:
        df = pd.read_csv(file_path, na_values=[], keep_default_na=False)
    elif dwell_time_logger is not None:
        df = dwell_time_logger
    else:
        print("No data provided.")
        return

    # Prepare data for Sankey plot
    source = df['state_name']
    target = df['going_to_state']
    value = df['dwell_time']

    # Create a dictionary to map unique labels to indices
    labels = list(set(source) | set(target))
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Map source and target to indices
    source_indices = source.map(label_to_index)
    target_indices = target.map(label_to_index)

    # Create the Sankey plot
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="gray", width=0.5),
            label=labels
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=value
        )
    ))

    fig.update_layout(title_text="Sankey Diagram of State Transitions and Dwell Times", font_size=10)
    fig.show()

# looks good /
def interactive_all_state_transitions(dwell_time_logger=None, dwell_time_bins=None, filter_states=None):
    """
    Plot the state transitions and/or dwell time distributions of agents interactively,
    with dwell times grouped into predefined ranges.

    Parameters:
    - dwell_time_bins (list): List of bin edges for grouping dwell times.
                                Default is [0, 50, 100, 150, 200, 250, np.inf].
    - filter_states (list): List of states to include in the plot. If None, include all states.
    """
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    if dwell_time_logger is None or dwell_time_logger.empty or 'dwell_time' not in dwell_time_logger.columns:
        print("No dwell time data available to plot.")
        return

    # Set default bins if none are provided
    if dwell_time_bins is None:
        dwell_time_bins = [0, 50, 100, 150, 200, 250, np.inf]

    # Create bin labels, handling infinity separately
    dwell_time_labels = [
        f"{int(b)}-{int(d)} days" if d != np.inf else f"{int(b)}+ days"
        for b, d in zip(dwell_time_bins[:-1], dwell_time_bins[1:])
    ]

    # Create a dwell time category column
    dwell_time_logger['dwell_time_category'] = pd.cut(
        dwell_time_logger['dwell_time'],
        bins=dwell_time_bins,
        labels=dwell_time_labels,
        include_lowest=True
    )

    # Apply state filter if provided
    if filter_states is not None:
        filtered_logger = dwell_time_logger[
            dwell_time_logger['state_name'].isin(filter_states) |
            dwell_time_logger['going_to_state'].isin(filter_states)
        ]
    else:
        filtered_logger = dwell_time_logger

    # Group by state transitions and dwell time category
    grouped = filtered_logger.groupby(
        ['state_name', 'going_to_state', 'dwell_time_category']
    ).size().reset_index(name='count')

    # Filter out empty ranges
    grouped = grouped[grouped['count'] > 0]

    # Interactive state transitions
    fig = go.Figure()

    for _, row in grouped.iterrows():
        state_label = row['state_name']
        going_to_state_label = row['going_to_state']
        dwell_time_category = row['dwell_time_category']
        count = row['count']
        unique_transition = f"{state_label} → {going_to_state_label} ({dwell_time_category})"

        # Add transition as a bar
        fig.add_trace(go.Bar(
            y=[unique_transition],
            x=[count],
            name=unique_transition,
            text=[count],
            textposition='auto',
            orientation='h'
        ))

    fig.update_layout(
        title="State Transitions Grouped by Dwell Time Categories",
        yaxis_title="State Transitions",
        xaxis_title="Count",
        legend_title="Transitions",
        height=100 + 30 * len(grouped),
    )
    fig.show()

# looks good - although crowded /
def stacked_bars_states_per_agent_static(dwell_time_logger=None, file_path=None):

    if file_path is not None:
        df = pd.read_csv(file_path, na_values=[], keep_default_na=False)
    elif dwell_time_logger is not None:
        df = dwell_time_logger
    else:
        print("No data provided.")
        return

    # Calculate cumulative dwell time for each agent and state
    df['cumulative_dwell_time'] = df.groupby(['agent_id', 'state_name'])['dwell_time'].cumsum()

    # Convert dwell time to days
    df['cumulative_dwell_time_days'] = df['cumulative_dwell_time']#/24

    # Pivot the data to get cumulative dwell time for each state
    pivot_df = df.pivot_table(index='agent_id', columns='state_name', values='cumulative_dwell_time_days', aggfunc='max', fill_value=0)

    # Plot the data
    pivot_df.plot(kind='bar', stacked=True, figsize=(15, 7))
    plt.title('Cumulative Time in Days on Each State for All Agents')
    plt.xlabel('Agent ID')
    plt.ylabel('Cumulative Time (Days)')
    plt.legend(title='State Name')
    plt.tight_layout()
    plt.show()
    return

# looks good
def interactive_stacked_bar_charts_dt_by_state(dwell_time_logger=None, bin_size=50, num_bins=20):
    """
    Generates an interactive stacked bar chart of dwell times by state using Plotly.

    Parameters:
    bin_size (int): The size of each bin for dwell times. Default is 50.
    num_bins (int): The number of bins to divide the dwell times into. Default is 20.
    dwell_time_logger (DataFrame): A pandas DataFrame containing dwell time data with columns 'state_name', 'dwell_time', and 'going_to_state'.
    
    Returns:
    None: Displays an interactive Plotly figure.
    
    Notes:
    - If the dwell_time_logger DataFrame is empty, the function will print a message and return without plotting.
    - The function creates bins for dwell times and labels them in days.
    - It generates a stacked bar chart for each state, showing the count of transitions to other states within each dwell time bin.
    - The height of the figure is dynamically adjusted based on the number of states.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    if dwell_time_logger.empty:
        print("No dwell time data available to plot.")
        return

    # Define bins for dwell times
    bins = np.arange(0, bin_size * num_bins, bin_size)
    bin_labels = [f"{int(b)}-{int(b + bin_size)} days" for b in bins[:-1]]

    # Create a figure with subplots for each state
    states = dwell_time_logger['state_name'].unique()
    num_states = len(states)
    fig = go.Figure()

    for state in states:
        state_data = dwell_time_logger[dwell_time_logger['state_name'] == state]
        state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

        # Group by dwell time bins and going to state
        grouped = state_data.groupby(['dwell_time_bin', 'going_to_state']).size().unstack(fill_value=0)

        for going_to_state in grouped.columns:
            fig.add_trace(go.Bar(
                x=grouped.index,
                y=grouped[going_to_state],
                name=f"{state} to {going_to_state}",
                text=grouped[going_to_state],
                textposition='auto'
            ))

    fig.update_layout(
        barmode='stack',
        title="Stacked Bar Charts of Dwell Times by State",
        xaxis_title="Dwell Time Bins",
        yaxis_title="Count",
        legend_title="State Transitions",
        height=400 + 50 * num_states
    )
    fig.show()

# looks good
def plot_state_transition_lengths_custom(dwell_time_logger=None, transitions_dict=None):
    """
    Plots the cumulative distribution of dwell times for different state transitions.

    This function generates a plot with individual lines representing the cumulative 
    distribution of dwell times for each specified state transition. If no transitions 
    dictionary is provided, a default one is used.

    Parameters:
    dwell_time_logger (pandas.DataFrame): A DataFrame containing the dwell time data. 
        It should have columns 'state_name', 'going_to_state', and 'dwell_time'.
    transitions_dict (dict): A dictionary where keys are state names and values are 
        lists of states to which transitions are considered. If None, a default 
        dictionary is used.

        i.e.:

        transitions_dict = {
            'None': ['Latent Slow', 'Latent Fast'],
            'Active Presymp': ['Active Smpos', 'Active Smneg', 'Active Exptb'],
        }
    Returns:
    None: The function displays the plot and does not return any value.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    if dwell_time_logger is None or dwell_time_logger.empty:
        print("No dwell time data available to plot.")
        return

    if transitions_dict is None:
        transitions_dict = {
            'None': ['Latent Slow', 'Latent Fast'],
            'Active Presymp': ['Active Smpos', 'Active Smneg', 'Active Exptb'],
        }

    # Create subplots
    num_plots = len(transitions_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    if num_plots == 1:
        axes = [axes]

    for ax, (state_name, going_to_states) in zip(axes, transitions_dict.items()):
        for transition in going_to_states:
            data = dwell_time_logger[
                (dwell_time_logger['state_name'] == state_name) &
                (dwell_time_logger['going_to_state'] == transition)
            ]['dwell_time']
            ax.plot(np.sort(data), np.linspace(0, 1, len(data)), label=f"{state_name} -> {transition}")
        ax.set_title(f"Transitions from {state_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Distribution")
        ax.legend()
    plt.tight_layout()
    plt.show()

# looks good /
def plot_binned_by_compartment(dwell_time_logger=None,  bin_size=50, num_bins=8):
    """
    Plot stacked bar charts for each state showing the distribution of dwell times in configurable bins.

    Parameters:
    - dwell_time_logger (pd.DataFrame): DataFrame containing dwell time data with columns 'state_name', 'dwell_time', and 'going_to_state'.
    - bin_size (int): Size of each bin for grouping dwell times. Default is 50 days.
    - num_bins (int): Number of bins to divide the dwell times into. Default is 10 bins.
    """
    import matplotlib.pyplot as plt

    if dwell_time_logger.empty:
        print("No dwell time data available to plot.")
        return

    # Define bins for dwell times
    bins = np.arange(0, bin_size*num_bins, bin_size)
    bin_labels = [f"{int(b)}-{int(b+bin_size)} days" for b in bins[:-1]]

    # Create a figure with subplots for each state
    states = dwell_time_logger['state_name'].unique()

    num_states = len(states)
    num_cols = 4
    num_rows = (num_states + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=True)
    fig.suptitle(f'State - Compartment Transitions)', fontsize=16)
    axes = axes.flatten()

    for ax, state in zip(axes, states):
        state_data = dwell_time_logger[dwell_time_logger['state_name'] == state]
        state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

        # Group by dwell time bins and going to state
        grouped = state_data.groupby(['dwell_time_bin', 'compartment']).size().unstack(fill_value=0)

        # Plot stacked bar chart
        grouped.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_title(f'State: {state}')
        ax.set_xlabel('Dwell Time Bins')
        ax.set_ylabel('Count')
        ax.legend(title='compartment')

    # Remove any empty subplots
    for i in range(num_states, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# looks good /
def plot_binned_stacked_bars_state_transitions(dwell_time_logger, bin_size=50, num_bins=8):
    """
    Plot stacked bar charts for each state showing the distribution of dwell times in configurable bins.

    Parameters:
    - dwell_time_logger (pd.DataFrame): DataFrame containing dwell time data with columns 'state_name', 'dwell_time', and 'going_to_state'.
    - bin_size (int): Size of each bin for grouping dwell times. Default is 50 days.
    - num_bins (int): Number of bins to divide the dwell times into. Default is 10 bins.
    """
    import matplotlib.pyplot as plt

    if dwell_time_logger.empty:
        print("No dwell time data available to plot.")
        return

    # Define bins for dwell times
    bins = np.arange(0, bin_size*num_bins, bin_size)
    bin_labels = [f"{int(b)}-{int(b+bin_size)} days" for b in bins[:-1]]

    # Create a figure with subplots for each state
    states = dwell_time_logger['state_name'].unique()
    num_states = len(states)
    num_cols = 4
    num_rows = (num_states + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=True)

    axes = axes.flatten()
    fig.suptitle(f'State Transitions by Dwell Time Bins)', fontsize=16)
    for ax, state in zip(axes, states):
        state_data = dwell_time_logger[dwell_time_logger['state_name'] == state]
        state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

        # Group by dwell time bins and going to state
        grouped = state_data.groupby(['dwell_time_bin', 'going_to_state']).size().unstack(fill_value=0)

        # Plot stacked bar chart
        grouped.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_title(f'State: {state}')
        ax.set_xlabel('Dwell Time Bins')
        ax.set_ylabel('Count')
        ax.legend(title='Going to State')

    # Remove any empty subplots
    for i in range(num_states, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# looks good /
def graph_state_transitions(dwell_time_logger=None, states=None, pos=None):
    """
    Plot a state transition graph with mean and mode dwell times annotated on the edges.

    Parameters:
    dwell_time_logger (pd.DataFrame): A DataFrame containing columns 'state_name', 'going_to_state', and 'dwell_time'.
                                      This DataFrame logs the dwell times for state transitions.
    states (list, optional): A list of states to include in the graph. If None, all states in the dwell_time_logger will be included.
    Returns:
    None: This function does not return any value. It displays a plot of the state transition graph.
    Notes:
    - The function uses NetworkX to create a directed graph where nodes represent states and edges represent transitions.
    - Each edge is annotated with the mean and mode dwell times, as well as the number of agents that made the transition.
    - If the dwell_time_logger is empty, the function prints a message and returns without plotting.
    - The graph layout is generated using a spring layout for better visualization.
    - Nodes are colored using a colormap, and edges are drawn with arrows to indicate direction.
    - The graph is displayed using Matplotlib.
    """
    import networkx as nx
    import itertools as it
    from scipy import stats

    if dwell_time_logger.empty:
        print("No data available to plot.")
        return

    # Calculate mean, mode, and count for each state transition
    transitions = dwell_time_logger.groupby(['state_name', 'going_to_state'])['dwell_time']
    stats_df = transitions.agg([
        'mean',
        lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) > 0 else np.nan,
        'count'
    ]).reset_index()

    stats_df.columns = ['state_name', 'going_to_state', 'mean', 'mode', 'count']

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with mean and mode annotations
    for _, row in stats_df.iterrows():
        from_state = row['state_name']
        to_state = row['going_to_state']
        mean_dwell = round(row['mean'], 2) if not pd.isna(row['mean']) else "N/A"
        mode_dwell = round(row['mode'], 2) if not pd.isna(row['mode']) else "N/A"
        num_agents = row['count']

        # Add edge to the graph
        G.add_edge(from_state, to_state,
                label=f"Mean: {mean_dwell}\nMode: {mode_dwell}\nAgents: {num_agents}")

    # Generate a layout for the graph
    if pos is None:
        pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
    else:
        pos = select_graph_pos(G, pos)

    colors = plt.cm.get_cmap('tab20', len(G.nodes))
    node_colors = [colors(i) for i in range(len(G.nodes))]
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_colors, alpha=0.9)
    
    # Draw edges with the same color as the origin node
    edge_colors = [node_colors[list(G.nodes).index(edge[0])] for edge in G.edges]
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=30, edge_color=edge_colors) #, connectionstyle="arc3,rad=0.1")
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")

    # Annotate edges with mean and mode
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Display the graph
    plt.tight_layout()
    plt.title("State Transition Graph with Dwell Times")
    plt.show()
    return


# Looks good /
def graph_compartments_transitions(dwell_time_logger=None, states=None, pos=0):
    """
    Plots a directed graph of state transitions with dwell times.

    Parameters:
    dwell_time_logger (DataFrame): A pandas DataFrame containing columns 'state_name', 'compartment', and 'dwell_time'.
                                   This DataFrame logs the dwell times for each state transition.
    states (list, optional): A list of state names to filter the dwell_time_logger. If None, all states are included.

    Returns:
    None: The function displays a plot of the state transition graph with annotations for mean, mode, and count of dwell times.
    """

    import networkx as nx
    import itertools as it
    from scipy import stats

    if dwell_time_logger.empty:
        print("No data available to plot.")
        return

    if states is not None:
        dwell_time_logger = dwell_time_logger[dwell_time_logger['state_name'].isin(states)]

    # Calculate mean, mode, and count for each state transition
    transitions = dwell_time_logger.groupby(['state_name', 'compartment'])['dwell_time']
    stats_df = transitions.agg([
        'mean',
        lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) > 0 else np.nan,
        'count'
    ]).reset_index()

    stats_df.columns = ['state_name', 'compartment', 'mean', 'mode', 'count']

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with mean and mode annotations
    for _, row in stats_df.iterrows():
        from_state = row['state_name']
        to_compartment = row['compartment']
        mean_dwell = round(row['mean'], 2) if not pd.isna(row['mean']) else "N/A"
        mode_dwell = round(row['mode'], 2) if not pd.isna(row['mode']) else "N/A"
        num_agents = row['count']

        # Add edge to the graph
        G.add_edge(from_state, to_compartment,
                    label=f"Mean: {mean_dwell}, Mode: {mode_dwell}\nAgents: {num_agents}")

    # Generate a layout for the graph
    pos = select_graph_pos(G, pos)

    # Draw nodes and edges with curved lines
    colors = plt.cm.get_cmap('tab20', len(G.nodes))
    node_colors = [colors(i) for i in range(len(G.nodes))]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=10, edge_color="black", connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")

    # Annotate edges with mean and mode
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # Display the graph
    plt.title("State->Compartment Graph with Dwell Times")
    plt.show()
    return

def select_graph_pos(G, pos, states=None):
    import networkx as nx
    if pos == 1: pos = nx.circular_layout(G)
    elif pos == 2: pos = nx.spiral_layout(G)
    elif pos == 3: pos = nx.spectral_layout(G)
    elif pos == 4: pos = nx.shell_layout(G)
    elif pos == 5: pos = nx.kamada_kawai_layout(G)
    elif pos == 6: pos = nx.planar_layout(G)
    elif pos == 7: pos = nx.random_layout(G)
    elif pos == 8: pos = nx.bipartite_layout(G, states)
    elif pos == 9: pos = nx.fruchterman_reingold_layout(G)
    else: pos = nx.spring_layout(G, seed=42)
    return pos


if __name__ == "__main__":
    file = f'/Users/mine/git/tbsim/results/dwell_time_logger_20250123160047.csv'

    # Load the dwell time logger
    dwell_time_logger = pd.read_csv(file, na_values=[], keep_default_na=False)
    graph_state_transitions(dwell_time_logger=dwell_time_logger )
    # stacked_bars_states_per_agent_static(file)
    # plot_kaplan_meier(dwell_time_logger, dwell_time_col='dwell_time')
    # transitions_dict = {
    #     'None': ['Latent Slow', 'Latent Fast'],
    #     'Active Presymp': ['Active Smpos', 'Active Smneg', 'Active Exptb'],
    # }    
    # plot_state_transition_lengths_custom(dwell_time_logger=dwell_time_logger, transitions_dict=transitions_dict)    

    # # sankey(dwell_time_logger=dwell_time_logger)
    # # state_transition_matrix(dwell_time_logger=dwell_time_logger)
    # interactive_stacked_bar_charts_dt_by_state(dwell_time_logger=dwell_time_logger, bin_size=50)
    # plot_state_transition_lengths_custom(dwell_time_logger=dwell_time_logger, transitions_dict=transitions_dict)
    # graph_state_transitions(dwell_time_logger=dwell_time_logger, states=['None', 'Latent Slow', 'Latent Fast', 'Active Presymp', 'Active Smpos', 'Active Smneg', 'Active Exptb'], pos=0 )
    # graph_compartments_transitions(dwell_time_logger=dwell_time_logger, states=['None', 'Active Presymp', 
    # plot_binned_by_compartment(dwell_time_logger=dwell_time_logger,  bin_size=50, num_bins=8)
    # plot_binned_stacked_bars_state_transitions(dwell_time_logger=dwell_time_logger, bin_size=50, num_bins=8)

