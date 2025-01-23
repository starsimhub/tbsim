import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go



def state_transition_matrix(file_path):

    df = pd.read_csv(file_path)
    # Create a transition matrix
    # Get the unique states
    unique_states = sorted(df['state'].dropna().unique())

    # Initialize a matrix of zeros
    transition_matrix = pd.DataFrame(
        data=0, index=unique_states, columns=unique_states, dtype=int
    )

    # Fill the matrix with transitions
    for agent_id, group in df.groupby('agent_id'):
        states = group['state'].values
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




def parallel_coordinates(file_path):
    import pandas as pd
    import plotly.express as px
    from ipywidgets import interact, Checkbox

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Function to create the plot based on selected dimensions
    def create_plot(**kwargs):
        selected_columns = [col for col, selected in kwargs.items() if selected]
        if not selected_columns:
            print("Please select at least one dimension.")
            return
        fig = px.parallel_coordinates(df, color='state', 
                                      dimensions=selected_columns,
                                      color_continuous_scale=px.colors.diverging.Tealrose,
                                      color_continuous_midpoint=2)
        fig.show()

    # Create checkboxes for each column
    checkboxes = {col: Checkbox(value=True, description=col) for col in df.columns}

    # Use interact to create the plot based on selected checkboxes
    interact(create_plot, **checkboxes)

def parallel_categories(file_path):
    import pandas as pd
    import plotly.express as px

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create a parallel categories plot
    fig = px.parallel_categories(df, dimensions=['agent_id', 'state', 'state_name'],
                                color='dwell_time', color_continuous_scale=px.colors.sequential.Inferno)

    # Show the plot
    fig.show()

def sankey(file_path):
    import plotly.graph_objects as go

    # Load the data

    df = pd.read_csv(file_path)

    # Prepare data for Sankey plot
    source = df['agent_id'].astype(str) + '-' + df['state'].astype(str)
    target = df['agent_id'].astype(str) + '-' + df['state'].shift(-1).fillna(df['state']).astype(str)
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
            line=dict(color="gray", width=0.3),
            label=labels
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=value
        )
    ))

    fig.update_layout(title_text="Sankey Diagram of Agent States and Dwell Times", font_size=10)
    fig.show()

def p3(file_path):

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Calculate cumulative dwell time for each state
    cumulative_dwell_time = df.groupby('state')['dwell_time'].sum()

    # Convert dwell time from hours to years (assuming dwell_time is in hours)
    cumulative_dwell_time_years = cumulative_dwell_time / (24 * 365)

    # Plot the cumulative dwell time for each state
    plt.figure(figsize=(10, 6))
    cumulative_dwell_time_years.plot(kind='bar')
    plt.title('Cumulative Dwell Time in Years for Each State')
    plt.xlabel('State')
    plt.ylabel('Cumulative Dwell Time (Years)')
    plt.xticks(rotation=0)
    plt.show()



    import plotly.express as px

    # Calculate cumulative dwell time for each state in days
    cumulative_dwell_time_days = cumulative_dwell_time

    # Create an interactive plot using Plotly
    fig = px.bar(cumulative_dwell_time_days, 
                x=cumulative_dwell_time_days.index, 
                y=cumulative_dwell_time_days.values, 
                labels={'x': 'State', 'y': 'Cumulative Dwell Time (Days)'},
                title='Cumulative Dwell Time in Days for Each State')

    # Add dropdown to select states
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="All",
                        method="update",
                        args=[{"visible": [True] * len(cumulative_dwell_time_days)},
                            {"title": "Cumulative Dwell Time in Days for Each State"}]),
                    dict(label="Top 5",
                        method="update",
                        args=[{"visible": [True] * 5 + [False] * (len(cumulative_dwell_time_days) - 5)},
                            {"title": "Top 5 States by Cumulative Dwell Time"}]),
                    dict(label="Bottom 5",
                        method="update",
                        args=[{"visible": [False] * (len(cumulative_dwell_time_days) - 5) + [True] * 5},
                            {"title": "Bottom 5 States by Cumulative Dwell Time"}]),
                ]),
                direction="down",
            )
        ]
    )

    fig.show()


def stacked_bars_states_per_agent(file_path):
    import plotly.express as px
    # Load the CSV file
    df = pd.read_csv(file_path)
    # Prepare the data for Plotly
    df['cumulative_dwell_time_days'] = df.groupby(['agent_id', 'state'])['dwell_time'].cumsum() / 24
    pivot_df = df.pivot_table(index='agent_id', columns='state', values='cumulative_dwell_time_days', aggfunc='max', fill_value=0).reset_index()

    # Melt the DataFrame for Plotly
    melted_df = pivot_df.melt(id_vars='agent_id', var_name='state', value_name='cumulative_dwell_time_days')

    # Create the interactive plot
    fig = px.bar(melted_df, x='agent_id', y='cumulative_dwell_time_days', color='state', title='Cumulative Time in Days on Each State for All Agents', labels={'cumulative_dwell_time_days': 'Cumulative Time (Days)', 'agent_id': 'Agent ID'})

    # Update layout for better visualization
    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'}, legend_title_text='State')

    # Show the plot
    fig.show()

def stacked_bars_states_per_agent_clean(file_path):
    import plotly.express as px
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter out rows with empty agent_id
    df = df[df['agent_id'].notna()]
    
    # Prepare the data for Plotly
    df['cumulative_dwell_time_days'] = df.groupby(['agent_id', 'state_name'])['dwell_time'].cumsum() / 24
    pivot_df = df.pivot_table(index='agent_id', columns='state_name', values='cumulative_dwell_time_days', aggfunc='max', fill_value=0).reset_index()

    # Melt the DataFrame for Plotly
    melted_df = pivot_df.melt(id_vars='agent_id', var_name='state_name', value_name='cumulative_dwell_time_days')

    # Create the interactive plot
    fig = px.bar(melted_df, x='agent_id', y='cumulative_dwell_time_days', color='state_name', title='Cumulative Time in Days on Each State for All Agents', labels={'cumulative_dwell_time_days': 'Cumulative Time (Days)', 'agent_id': 'Agent ID'})

    # Update layout for better visualization
    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending', 'type': 'category'}, legend_title_text='State')

    # Show the plot
    fig.show()

def plot_dwell_time_lines_for_each_agent(file_path):

    # Load the CSV data
    data = pd.read_csv(file_path)

    # Display the first few rows of the data
    print(data.head())

    # Generate a plot
    plt.figure(figsize=(10, 6))
  
    # Plot dwell time for each agent
    for agent_id in data['agent_id'].unique():
        agent_data = data[data['agent_id'] == agent_id]
        plt.plot(agent_data['dwell_time'], label=f'Agent {agent_id}')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Dwell Time')
    plt.title('Dwell Time for Each Agent')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def interactive_plot_dwell_time_lines_for_each_agent(file_path):
    import plotly.express as px

    # Load the CSV data
    data = pd.read_csv(file_path)

    # Create an interactive line plot using Plotly
    fig = px.line(data, x=data.index, y='dwell_time', color='agent_id', title='Dwell Time for Each Agent')

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Index',
        yaxis_title='Dwell Time',
        legend_title_text='Agent ID'
    )

    # Show the plot
    fig.show()


def plot_dwell_time_lines_for_each_agent_fixed(file_path):

    # Load the CSV data
    data = pd.read_csv(file_path)

    # Display the first few rows of the data
    print(data.head())

    # Generate a plot
    plt.figure(figsize=(10, 6))
  
    # Plot dwell time for each agent
    for agent_id in data['agent_id'].unique():
        agent_data = data[data['agent_id'] == agent_id]
        plt.plot(agent_data['dwell_time'], label=f'Agent {agent_id}')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Dwell Time')
    plt.title('Dwell Time for Each Agent')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
def group_by_state(file_path):

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Group by state and count the occurrences
    state_counts = df['state'].value_counts().sort_index()

    # Plot the frequency of each state
    plt.figure(figsize=(10, 6))
    state_counts.plot(kind='bar')
    plt.xlabel('State')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each State')
    plt.xticks(rotation=0)
    plt.show()

def interactive_dwell_time_by_state(file_path):
    # Load the data with error handling
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path.")

    # Ensure the 'dwell_time', 'agent_id', and 'state' columns exist
    required_columns = {'dwell_time', 'agent_id', 'state'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The required columns {required_columns} are missing from the file.")

    # Convert dwell_time to days (adjust logic if needed)
    df['dwell_time_days'] = df['dwell_time']

    # Group by agent_id and state, summing dwell_time_days
    grouped_df = df.groupby(['agent_id', 'state'])['dwell_time_days'].sum().reset_index()

    # Get unique states and initialize the figure
    unique_states = grouped_df['state'].unique()
    fig = go.Figure()

    # Add a bar trace for each state
    for state in unique_states:
        state_df = grouped_df[grouped_df['state'] == state]
        fig.add_trace(
            go.Bar(
                x=state_df['agent_id'],
                y=state_df['dwell_time_days'],
                name=f"State {state}",
                visible=True  # All traces visible initially
            )
        )

    # Initialize visibility for all states
    visibility = [True] * len(unique_states)

    # Define buttons for each state to toggle visibility
    dropdown_buttons = [
        dict(
            label=f"Toggle State {state}",
            method="update",
            args=[
                {
                    "visible": [
                        not visibility[idx] if unique_states[idx] == state else visibility[idx]
                        for idx in range(len(unique_states))
                    ]
                },
                {"title": f"States Visible: {[s for s, v in zip(unique_states, visibility) if v]}"}  # Update title
            ],
        )
        for state in unique_states
    ]

    # Add a "Show All" button to reset visibility
    dropdown_buttons.insert(
        0,
        dict(
            label="Show All",
            method="update",
            args=[
                {"visible": [True] * len(unique_states)},
                {"title": "Cumulative Dwell Time in Days by State for Each Agent"}
            ],
        )
    )

    # Add a "Hide All" button to hide all traces
    dropdown_buttons.insert(
        1,
        dict(
            label="Hide All",
            method="update",
            args=[
                {"visible": [False] * len(unique_states)},
                {"title": "No States Visible"}
            ],
        )
    )

    # Update figure layout with dropdown menu
    fig.update_layout(
        title="Cumulative Dwell Time in Days by State for Each Agent",
        xaxis_title="Agent ID",
        yaxis_title="Cumulative Dwell Time (Days)",
        barmode="stack",
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,  # Position of the dropdown menu
                y=1.15,
            )
        ]
    )

    # Render the figure
    fig.show()

def cumulative_dwell_time_by_state(file_path):
    # Load the data with error handling
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path.")

    # Ensure the required columns exist
    required_columns = {'dwell_time', 'agent_id', 'state'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The required columns {required_columns} are missing from the file.")

    # Convert dwell_time to days (adjust logic if needed)
    df['dwell_time_days'] = df['dwell_time']

    # Group by agent_id and state, summing dwell_time_days
    grouped_df = df.groupby(['agent_id', 'state'])['dwell_time_days'].sum().reset_index()

    # Get unique states and initialize the figure
    unique_states = grouped_df['state'].unique()
    fig = go.Figure()

    # Add a bar trace for each state
    for state in unique_states:
        state_df = grouped_df[grouped_df['state'] == state]
        fig.add_trace(
            go.Bar(
                x=state_df['agent_id'],
                y=state_df['dwell_time_days'],
                name=f"State {state}",
                visible=True  # All traces visible initially
            )
        )

    # Create dropdown buttons for multi-selection
    dropdown_buttons = []
    for i, state in enumerate(unique_states):
        visibility_array = [True] * len(unique_states)  # Start with all traces visible
        visibility_array[i] = False  # Toggle visibility for the current state
        dropdown_buttons.append(
            dict(
                label=f"Toggle State {state}",
                method="update",
                args=[
                    {"visible": visibility_array},
                    {"title": f"Cumulative Dwell Time in Days (Toggle State {state})"}
                ]
            )
        )

    # Add a "Show All" button to display all states
    dropdown_buttons.insert(
        0,
        dict(
            label="Show All",
            method="update",
            args=[
                {"visible": [True] * len(unique_states)},
                {"title": "Cumulative Dwell Time in Days by State for Each Agent"}
            ]
        )
    )

    # Add a "Hide All" button to hide all states
    dropdown_buttons.insert(
        1,
        dict(
            label="Hide All",
            method="update",
            args=[
                {"visible": [False] * len(unique_states)},
                {"title": "Cumulative Dwell Time in Days (All States Hidden)"}
            ]
        )
    )

    # Update figure layout with dropdown menu
    fig.update_layout(
        title="Cumulative Dwell Time in Days by State for Each Agent",
        xaxis_title="Agent ID",
        yaxis_title="Cumulative Dwell Time (Days)",
        barmode="stack",
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,  # Position of the dropdown menu
                y=1.15,
            )
        ]
    )

    # Render the figure
    fig.show()


def cumulative_dwell_time_toggle(file_path):

    # Load the data with error handling
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path.")

    # Ensure the required columns exist
    required_columns = {'dwell_time', 'agent_id', 'state'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The required columns {required_columns} are missing from the file.")

    # Convert dwell_time to days (adjust logic if needed)
    df['dwell_time_days'] = df['dwell_time']

    # Group by agent_id and state, summing dwell_time_days
    grouped_df = df.groupby(['agent_id', 'state'])['dwell_time_days'].sum().reset_index()

    # Get unique states and initialize the figure
    unique_states = grouped_df['state'].unique()
    fig = go.Figure()

    # Add a bar trace for each state
    for state in unique_states:
        state_df = grouped_df[grouped_df['state'] == state]
        fig.add_trace(
            go.Bar(
                x=state_df['agent_id'],
                y=state_df['dwell_time_days'],
                name=f"State {state}",
                visible=True  # All traces visible initially
            )
        )

    # Initialize visibility list (all traces visible initially)
    visibility_list = [True] * len(unique_states)

    # Create dropdown buttons for toggling states
    dropdown_buttons = []

    for i, state in enumerate(unique_states):
        # Create a button to toggle the specific state
        dropdown_buttons.append(
            dict(
                label=f"Toggle State {state}",
                method="update",
                args=[
                    {"visible": [
                        not visibility_list[j] if j == i else visibility_list[j]
                        for j in range(len(unique_states))
                    ]},
                    {"title": "Cumulative Dwell Time in Days by Selected States"}
                ]
            )
        )

    # Add a "Reset All" button to display all states
    dropdown_buttons.insert(
        0,
        dict(
            label="Reset All",
            method="update",
            args=[
                {"visible": [True] * len(unique_states)},
                {"title": "Cumulative Dwell Time in Days by State for Each Agent"}
            ]
        )
    )

    # Update figure layout with dropdown menu
    fig.update_layout(
        title="Cumulative Dwell Time in Days by State for Each Agent",
        xaxis_title="Agent ID",
        yaxis_title="Cumulative Dwell Time (Days)",
        barmode="stack",
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,  # Position of the dropdown menu
                y=1.15,
            )
        ]
    )

    # Render the figure
    fig.show()

def cumulative_dwell_time_pd8(file_path):

    # Load the data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path.")

    # Ensure required columns exist
    required_columns = {'dwell_time', 'agent_id', 'state'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The required columns {required_columns} are missing from the file.")

    # Prepare the data
    df['dwell_time_days'] = df['dwell_time']/30
    grouped_df = df.groupby(['agent_id', 'state'])['dwell_time_days'].sum().reset_index()
    unique_states = grouped_df['state'].unique()

    # Create traces for each state
    traces = []
    for state in unique_states:
        state_df = grouped_df[grouped_df['state'] == state]
        traces.append(
            go.Bar(
                x=state_df['agent_id'],
                y=state_df['dwell_time_days'],
                name=f"State {state}",
                visible=True  # Initially visible
            )
        )

    # Create layout and dropdown buttons
    buttons = []

    # Add 'Show All' button
    buttons.append(
        dict(
            label="Show PD 8 All",
            method="update",
            args=[
                {"visible": [True] * len(unique_states)},  # Show all traces
                {"title": "Cumulative Dwell Time in Days by State for Each Agent"}
            ]
        )
    )

    # Add 'Hide All' button
    buttons.append(
        dict(
            label="Hide All",
            method="update",
            args=[
                {"visible": [False] * len(unique_states)},  # Hide all traces
                {"title": "Cumulative Dwell Time (No States Visible)"}
            ]
        )
    )

    # Add a button for each state to toggle visibility
    for i, state in enumerate(unique_states):
        st = str(state)
        visibility = [trace.name.endswith(st) for trace in traces]
        buttons.append(
            dict(
                label=f"Toggle {state}",
                method="update",
                args=[
                    {"visible": [not vis if j == i else vis for j, vis in enumerate(visibility)]},
                    {"title": f"modified Cumulative Dwell Time (State {state} Toggled)"}
                ]
            )
        )

    # Build the figure
    fig = go.Figure(data=traces)

    fig.update_layout(
        title="Cumulative Dwell Time ooooo in Days by State for Each Agent",
        xaxis_title="Agent ID",
        yaxis_title="Cumulative Dwell Time (Days)",
        barmode="stack",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
            )
        ]
    )

    # Show the plot
    fig.show()



def stacked_bars_states_per_agent_static(file_path):

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Calculate cumulative dwell time for each agent and state
    df['cumulative_dwell_time'] = df.groupby(['agent_id', 'state'])['dwell_time'].cumsum()

    # Convert dwell time to days
    df['cumulative_dwell_time_days'] = df['cumulative_dwell_time']/24

    # Pivot the data to get cumulative dwell time for each state
    pivot_df = df.pivot_table(index='agent_id', columns='state', values='cumulative_dwell_time_days', aggfunc='max', fill_value=0)

    # Plot the data
    pivot_df.plot(kind='bar', stacked=True, figsize=(15, 7))
    plt.title('Cumulative Time in Days on Each State for All Agents')
    plt.xlabel('Agent ID')
    plt.ylabel('Cumulative Time (Days)')
    plt.legend(title='State')
    plt.tight_layout()
    plt.show()

def plot_stacked_bars_by_state_interactive(self, bin_size=50, dwell_time_logger=None):
    """
    Plot stacked bar charts for each state showing the distribution of dwell times in configurable bins interactively using Plotly.

    Parameters:
    - bin_size (int): Size of each bin for grouping dwell times. Default is 50 days.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    if dwell_time_logger.empty:
        print("No dwell time data available to plot.")
        return

    # Define bins for dwell times
    bins = np.arange(0, bin_size * 8, bin_size)
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

def plot_combined_rates_individual_lines(dwell_time_logger=None):

    import matplotlib.pyplot as plt
    import numpy as np

    if dwell_time_logger is None:
        dwell_time_logger = dwell_time_logger

    if dwell_time_logger.empty:
        print("No dwell time data available to plot.")
        return

    latent_transitions = dwell_time_logger[
        (dwell_time_logger['state_name'] == 'None') &
        (dwell_time_logger['going_to_state'].isin(['Latent Slow', 'Latent Fast']))
    ]

    active_transitions = dwell_time_logger[
        (dwell_time_logger['state_name']=='Active Presymp') &
        (dwell_time_logger['going_to_state'].isin(['Active Smpos', 'Active Smneg', 'Active Exptb']))
    ]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for None -> Latent
    for transition in ['Latent Slow', 'Latent Fast']:
        data = latent_transitions[latent_transitions['going_to_state'] == transition]['dwell_time']
        axes[0].plot(np.sort(data), np.linspace(0, 1, len(data)), label=f"None -> {transition}")
    axes[0].set_title("Latent Transitions")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Cumulative Distribution")
    axes[0].legend()

    # Plot for Active Presym
    for transition in ['Active Smpos', 'Active Smneg', 'Active Exptb']:
        data = active_transitions[active_transitions['going_to_state'] == transition]['dwell_time']
        axes[1].plot(np.sort(data), np.linspace(0, 1, len(data)), label=f"Active Presymp -> {transition}")
    axes[1].set_title("Active Presym Transitions")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Cumulative Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# ---------------------------- verified ---------------
# looks good
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

# looks good
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

# looks good
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
                label=f"Mean: {mean_dwell}, Mode: {mode_dwell}\nAgents: {num_agents}")

    # Generate a layout for the graph
    pos = select_graph_pos(G, pos)

    # Draw nodes and edges with curved lines
    colors = plt.cm.get_cmap('tab20', len(G.nodes))
    node_colors = [colors(i) for i in range(len(G.nodes))]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=10, edge_color="black") #connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")

    # Annotate edges with mean and mode
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Display the graph
    plt.title("State Transition Graph with Dwell Times")
    plt.show()
    return

# Looks good
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
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=10, edge_color="black")  # connectionstyle="arc3,rad=0.2")
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
    
    file = f'/Users/mine/git/tbsim/tbsim/results/dwell_time_logger_20250122185948.csv'

    # Load the dwell time logger
    dwell_time_logger = pd.read_csv(file, na_values=[], keep_default_na=False)

    plot_combined_rates_individual_lines(dwell_time_logger=dwell_time_logger)

    # graph_state_transitions(dwell_time_logger=dwell_time_logger, states=['None', 'Latent Slow', 'Latent Fast', 'Active Presymp', 'Active Smpos', 'Active Smneg', 'Active Exptb'], pos=0 )
    # graph_compartments_transitions(dwell_time_logger=dwell_time_logger, states=['None', 'Active Presymp', 
    #                                                                             'Active Smpos', 'Active Smneg', 'Active Exptb'], pos=4)
    # plot_binned_by_compartment(dwell_time_logger=dwell_time_logger,  bin_size=50, num_bins=8)
    # plot_binned_stacked_bars_state_transitions(dwell_time_logger=dwell_time_logger, bin_size=50, num_bins=8)
    
    pass