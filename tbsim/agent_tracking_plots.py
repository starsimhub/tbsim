"""
Agent State Tracking Visualization Module

This module provides comprehensive visualization functions for analyzing
agent state progression patterns in the TB state machine.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from collections import defaultdict, Counter


def plot_agent_state_paths(state_manager, max_agents: int = 20, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot individual agent state progression paths over time.
    
    Args:
        state_manager: TBStateManager instance with agent tracking data
        max_agents: Maximum number of agent paths to display
        figsize: Figure size tuple
    """
    agent_histories = state_manager.get_all_agent_histories()
    
    if not agent_histories:
        print("No agent histories available for plotting.")
        return
    
    # Select agents with the most transitions for visualization
    agent_transition_counts = {uid: len(history) for uid, history in agent_histories.items()}
    top_agents = sorted(agent_transition_counts.items(), key=lambda x: x[1], reverse=True)[:max_agents]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for each state
    state_colors = {
        'Clear': '#2E8B57',           # Sea Green
        'Latent Slow': '#4169E1',     # Royal Blue
        'Latent Fast': '#1E90FF',     # Dodger Blue
        'Active Pre-symptomatic': '#FFD700',  # Gold
        'Active Smear Positive': '#DC143C',   # Crimson
        'Active Smear Negative': '#FF6347',   # Tomato
        'Active Extra-pulmonary': '#8A2BE2',  # Blue Violet
        'Dead': '#696969',            # Dim Gray
        'Protected': '#32CD32'        # Lime Green
    }
    
    # Plot each agent's path
    for i, (uid, _) in enumerate(top_agents):
        history = agent_histories[uid]
        if len(history) < 2:
            continue
            
        times = [entry[0] for entry in history]
        states = [entry[2] for entry in history]
        
        # Create line segments for each transition
        for j in range(len(times) - 1):
            color = state_colors.get(states[j], '#808080')
            ax.plot([times[j], times[j+1]], [i, i], color=color, linewidth=3, alpha=0.7)
            
            # Add transition arrow
            if j < len(times) - 1:
                ax.annotate('', xy=(times[j+1], i), xytext=(times[j], i),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
    
    # Customize plot
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Agent ID', fontsize=12)
    ax.set_title(f'Agent State Progression Paths (Top {len(top_agents)} Agents)', fontsize=14, fontweight='bold')
    
    # Set y-axis labels
    ax.set_yticks(range(len(top_agents)))
    ax.set_yticklabels([f'Agent {uid}' for uid, _ in top_agents])
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=3, label=state) 
                      for state, color in state_colors.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()


def plot_state_duration_distribution(state_manager, figsize: Tuple[int, int] = (12, 8)):
    """
    Plot distribution of time spent in each state.
    
    Args:
        state_manager: TBStateManager instance with agent tracking data
        figsize: Figure size tuple
    """
    progression_stats = state_manager.get_state_progression_stats()
    
    # Prepare data for plotting
    states = []
    durations = []
    
    for state_name, stats in progression_stats.items():
        if stats['durations']:
            states.extend([state_name] * len(stats['durations']))
            durations.extend(stats['durations'])
    
    if not durations:
        print("No duration data available for plotting.")
        return
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({'State': states, 'Duration': durations})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    sns.boxplot(data=df, x='State', y='Duration', ax=ax1)
    ax1.set_title('State Duration Distribution (Box Plot)', fontweight='bold')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Duration (days)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(data=df, x='State', y='Duration', ax=ax2)
    ax2.set_title('State Duration Distribution (Violin Plot)', fontweight='bold')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Duration (days)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_transition_flow_network(state_manager, figsize: Tuple[int, int] = (12, 10)):
    """
    Create a network diagram showing transition flows between states.
    
    Args:
        state_manager: TBStateManager instance with agent tracking data
        figsize: Figure size tuple
    """
    flow_stats = state_manager.get_transition_flow_analysis()
    
    if not flow_stats:
        print("No transition flow data available for plotting.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all unique states
    all_states = set()
    for (from_state, to_state) in flow_stats.keys():
        all_states.add(from_state)
        all_states.add(to_state)
    
    # Create state positions in a circle
    n_states = len(all_states)
    angles = np.linspace(0, 2*np.pi, n_states, endpoint=False)
    state_positions = {}
    
    for i, state in enumerate(all_states):
        x = np.cos(angles[i])
        y = np.sin(angles[i])
        state_positions[state] = (x, y)
    
    # Draw state nodes
    for state, (x, y) in state_positions.items():
        circle = plt.Circle((x, y), 0.1, color='lightblue', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, state.replace(' ', '\n'), ha='center', va='center', 
                fontsize=8, fontweight='bold')
    
    # Draw transition arrows
    max_flow = max(flow_stats[key]['count'] for key in flow_stats)
    
    for (from_state, to_state), stats in flow_stats.items():
        if stats['count'] > 0:
            x1, y1 = state_positions[from_state]
            x2, y2 = state_positions[to_state]
            
            # Calculate arrow properties based on flow strength
            width = 0.5 + 3 * (stats['count'] / max_flow)
            alpha = 0.3 + 0.7 * (stats['count'] / max_flow)
            
            # Draw arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=width, alpha=alpha, color='red'))
            
            # Add flow count label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, str(stats['count']), ha='center', va='center',
                   fontsize=8, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('TB State Transition Flow Network', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_state_occupancy_heatmap(state_manager, time_bins: int = 50, figsize: Tuple[int, int] = (15, 8)):
    """
    Create a heatmap showing state occupancy over time.
    
    Args:
        state_manager: TBStateManager instance with agent tracking data
        time_bins: Number of time bins for the heatmap
        figsize: Figure size tuple
    """
    agent_histories = state_manager.get_all_agent_histories()
    
    if not agent_histories:
        print("No agent histories available for plotting.")
        return
    
    # Get time range
    all_times = []
    for history in agent_histories.values():
        all_times.extend([entry[0] for entry in history])
    
    if not all_times:
        print("No time data available for plotting.")
        return
    
    min_time, max_time = min(all_times), max(all_times)
    time_bin_edges = np.linspace(min_time, max_time, time_bins + 1)
    
    # Get all unique states
    all_states = set()
    for history in agent_histories.values():
        all_states.update([entry[2] for entry in history])
    
    # Create occupancy matrix
    occupancy_matrix = np.zeros((len(all_states), time_bins))
    state_list = sorted(list(all_states))
    
    # Fill occupancy matrix
    for history in agent_histories.values():
        for i in range(len(history) - 1):
            time_start, state_start = history[i][0], history[i][2]
            time_end, state_end = history[i + 1][0], history[i + 1][2]
            
            # Find time bins for this period
            start_bin = np.digitize(time_start, time_bin_edges) - 1
            end_bin = np.digitize(time_end, time_bin_edges) - 1
            
            # Ensure bins are within bounds
            start_bin = max(0, min(start_bin, time_bins - 1))
            end_bin = max(0, min(end_bin, time_bins - 1))
            
            # Mark occupancy
            state_idx = state_list.index(state_start)
            occupancy_matrix[state_idx, start_bin:end_bin + 1] += 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(occupancy_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(0, time_bins, max(1, time_bins // 10)))
    ax.set_xticklabels([f'{time_bin_edges[i]:.0f}' for i in range(0, time_bins, max(1, time_bins // 10))])
    ax.set_yticks(range(len(state_list)))
    ax.set_yticklabels(state_list)
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title('State Occupancy Heatmap Over Time', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Agents', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_agent_progression_summary(state_manager, figsize: Tuple[int, int] = (15, 10)):
    """
    Create a comprehensive summary plot of agent progression patterns.
    
    Args:
        state_manager: TBStateManager instance with agent tracking data
        figsize: Figure size tuple
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. State entry counts
    progression_stats = state_manager.get_state_progression_stats()
    states = list(progression_stats.keys())
    entry_counts = [progression_stats[state]['total_entries'] for state in states]
    
    ax1.bar(states, entry_counts, color='skyblue', alpha=0.7)
    ax1.set_title('Total State Entries', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Average state durations
    avg_durations = [progression_stats[state]['avg_duration'] for state in states]
    ax2.bar(states, avg_durations, color='lightcoral', alpha=0.7)
    ax2.set_title('Average State Duration', fontweight='bold')
    ax2.set_ylabel('Duration (days)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Transition flow counts
    flow_stats = state_manager.get_transition_flow_analysis()
    if flow_stats:
        transitions = [f"{from_state} → {to_state}" for (from_state, to_state) in flow_stats.keys()]
        flow_counts = [flow_stats[key]['count'] for key in flow_stats.keys()]
        
        ax3.barh(transitions, flow_counts, color='lightgreen', alpha=0.7)
        ax3.set_title('Transition Flow Counts', fontweight='bold')
        ax3.set_xlabel('Count')
    else:
        ax3.text(0.5, 0.5, 'No transition data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Transition Flow Counts', fontweight='bold')
    
    # 4. Agent transition counts distribution
    agent_histories = state_manager.get_all_agent_histories()
    if agent_histories:
        transition_counts = [len(history) - 1 for history in agent_histories.values() if len(history) > 1]
        if transition_counts:
            ax4.hist(transition_counts, bins=20, color='gold', alpha=0.7, edgecolor='black')
            ax4.set_title('Distribution of Agent Transition Counts', fontweight='bold')
            ax4.set_xlabel('Number of Transitions')
            ax4.set_ylabel('Number of Agents')
        else:
            ax4.text(0.5, 0.5, 'No transition data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Distribution of Agent Transition Counts', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No agent data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Distribution of Agent Transition Counts', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def create_agent_tracking_report(state_manager, save_path: Optional[str] = None):
    """
    Create a comprehensive text report of agent tracking statistics.
    
    Args:
        state_manager: TBStateManager instance with agent tracking data
        save_path: Optional path to save the report
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("TB STATE MACHINE AGENT TRACKING REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Basic statistics
    agent_histories = state_manager.get_all_agent_histories()
    total_agents = len(agent_histories)
    total_transitions = len(state_manager.get_transition_log())
    
    report_lines.append(f"Total Agents Tracked: {total_agents}")
    report_lines.append(f"Total Transitions Recorded: {total_transitions}")
    report_lines.append("")
    
    # State progression statistics
    progression_stats = state_manager.get_state_progression_stats()
    report_lines.append("STATE PROGRESSION STATISTICS:")
    report_lines.append("-" * 40)
    
    for state_name, stats in progression_stats.items():
        report_lines.append(f"\n{state_name}:")
        report_lines.append(f"  Total Entries: {stats['total_entries']}")
        report_lines.append(f"  Total Exits: {stats['total_exits']}")
        report_lines.append(f"  Average Duration: {stats['avg_duration']:.2f} days")
        if stats['durations']:
            report_lines.append(f"  Min Duration: {min(stats['durations']):.2f} days")
            report_lines.append(f"  Max Duration: {max(stats['durations']):.2f} days")
    
    # Transition flow analysis
    flow_stats = state_manager.get_transition_flow_analysis()
    if flow_stats:
        report_lines.append("\n\nTRANSITION FLOW ANALYSIS:")
        report_lines.append("-" * 40)
        
        for (from_state, to_state), stats in flow_stats.items():
            report_lines.append(f"\n{from_state} → {to_state}:")
            report_lines.append(f"  Total Transitions: {stats['count']}")
            report_lines.append(f"  Unique Agents: {stats['unique_agents']}")
    
    # Agent-specific statistics
    report_lines.append("\n\nAGENT-SPECIFIC STATISTICS:")
    report_lines.append("-" * 40)
    
    if agent_histories:
        transition_counts = [len(history) - 1 for history in agent_histories.values()]
        if transition_counts:
            report_lines.append(f"Average Transitions per Agent: {np.mean(transition_counts):.2f}")
            report_lines.append(f"Max Transitions by Single Agent: {max(transition_counts)}")
            report_lines.append(f"Min Transitions by Single Agent: {min(transition_counts)}")
    
    report_lines.append("\n" + "=" * 60)
    
    # Join and print/save report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {save_path}")
    
    return report_text
