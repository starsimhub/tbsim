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

# def dwell_time_histogram(file_path):

#     df = pd.read_csv(file_path)

#     # Create a histogram of dwell times for each state
#     plt.figure(figsize=(12, 8))
#     sns.histplot(data=df, x='dwell_time', hue='state', bins=20, kde=True, palette='tab10', multiple='stack')

#     # Add title and labels
#     plt.title('Distribution of Dwell Times by State')
#     plt.xlabel('Dwell Time')
#     plt.ylabel('Frequency')

#     # Add legend
#     # plt.legend(title='State', loc='upper right')
#     plt.legend(title='State', loc='upper right', labels=sorted(df['state'].unique()))
#     plt.show()


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