"""Custom contact networks"""

import starsim as ss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

__all__ = ['plot_household_structure']


def plot_household_structure(households, title="Household Network Structure", figsize=(12, 8)):
    """
    Plot the structure of household networks showing connections within households.
    
    This function creates a network visualization where:
    - Nodes represent individual agents
    - Edges represent household connections (complete graphs within households)
    - Different colors represent different households
    - Node size is proportional to household size
    
    Args:
        households (list): List of lists, where each inner list contains agent UIDs in a household
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
        
    Returns:
        networkx.Graph: The NetworkX graph object for further analysis
        
    Example:
        >>> from tbsim.networks import plot_household_structure
        >>> households = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        >>> G = plot_household_structure(households, "My Household Network")
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all agents as nodes
    all_agents = [agent for hh in households for agent in hh]
    G.add_nodes_from(all_agents)
    
    # Add household membership as node attributes
    for hh_idx, household in enumerate(households):
        for agent in household:
            G.nodes[agent]['household'] = hh_idx
            G.nodes[agent]['household_size'] = len(household)
    
    # Add edges within households (complete graphs)
    for hh_idx, household in enumerate(households):
        if len(household) > 1:
            for i in range(len(household)):
                for j in range(i + 1, len(household)):
                    G.add_edge(household[i], household[j], household=hh_idx)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Generate colors for households
    household_colors = plt.cm.Set3(np.linspace(0, 1, len(households)))
    
    # Create layout with household clustering
    pos = {}
    if len(households) == 1:
        # Single household - use circular layout
        pos = nx.circular_layout(G)
    else:
        # Multiple households - arrange in circle with internal clustering
        household_angles = np.linspace(0, 2*np.pi, len(households), endpoint=False)
        
        for hh_idx, household in enumerate(households):
            # Base position for this household
            base_radius = 3.0
            hh_x = base_radius * np.cos(household_angles[hh_idx])
            hh_y = base_radius * np.sin(household_angles[hh_idx])
            
            if len(household) == 1:
                pos[household[0]] = (hh_x, hh_y)
            else:
                # Arrange household members in a small circle
                agent_angles = np.linspace(0, 2*np.pi, len(household), endpoint=False)
                inner_radius = 0.3 + 0.1 * len(household)
                
                for agent_idx, agent in enumerate(household):
                    pos[agent] = (hh_x + inner_radius * np.cos(agent_angles[agent_idx]),
                                 hh_y + inner_radius * np.sin(agent_angles[agent_idx]))
    
    # Draw network by household
    for hh_idx, household in enumerate(households):
        # Node sizes proportional to household size
        node_sizes = [300 + 50 * len(household) for _ in household]
        
        # Draw nodes for this household
        node_positions = {node: pos[node] for node in household}
        nx.draw_networkx_nodes(G, node_positions,
                             nodelist=household,
                             node_color=[household_colors[hh_idx]] * len(household),
                             node_size=node_sizes,
                             alpha=0.8,
                             edgecolors='black',
                             linewidths=2)
        
        # Draw edges for this household
        household_edges = [(u, v) for u, v in G.edges() if u in household and v in household]
        if household_edges:
            nx.draw_networkx_edges(G, pos,
                                 edgelist=household_edges,
                                 edge_color=household_colors[hh_idx],
                                 width=3,
                                 alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white')
    
    # Create legend
    legend_elements = []
    for hh_idx, household in enumerate(households):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=household_colors[hh_idx],
                                        markersize=12, alpha=0.8, markeredgecolor='black',
                                        label=f'Household {hh_idx + 1} (n={len(household)})'))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print network statistics
    total_agents = len(all_agents)
    total_edges = G.number_of_edges()
    avg_household_size = np.mean([len(hh) for hh in households])
    
    print(f"Network Statistics:")
    print(f"  Total agents: {total_agents}")
    print(f"  Total households: {len(households)}")
    print(f"  Total edges: {total_edges}")
    print(f"  Average household size: {avg_household_size:.1f}")
    print(f"  Household sizes: {[len(hh) for hh in households]}")
    
    return G
