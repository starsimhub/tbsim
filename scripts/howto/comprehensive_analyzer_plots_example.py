import tbsim as mtb
import starsim as ss
import starsim_examples as sse
import sciris as sc 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

TBS = mtb.TBS

def build_tbsim(sim_pars=None):
    """Build a TB simulation with dwell time analyzer"""
    sim_params = dict(
        start = ss.date('2013-01-01'),      
        stop = ss.date('2016-12-31'), 
        rand_seed=123,
        dt=ss.days(7),
    )
    if sim_pars is not None:
        sim_params.update(sim_pars)

    pop = ss.People(n_agents=1000)

    tb_params = dict(
        beta=ss.per(0.0025),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
        )
    tb = mtb.TB(tb_params)
    
    net = sse.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))

    dwell_analyzer = mtb.DwtAnalyzer(adjust_to_unit=True  # TODO: Check if adjust_to_unit is still needed in v3, dt=1.0, scenario_name='comprehensive_plots_example')

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        pars=sim_params,
        analyzers=dwell_analyzer,
    )
    sim.pars.verbose = 30/365

    return sim

def demonstrate_all_analyzer_plots():
    """Demonstrate all available plotting methods in the DwtAnalyzer class"""
    
    print("Building and running TB simulation...")
    sim_tb = build_tbsim()
    sim_tb.run()
    
    # Extract the analyzer
    analyzer = sim_tb.analyzers[0]
    
    print("\n=== COMPREHENSIVE DWT ANALYZER PLOTS DEMONSTRATION ===\n")
    
    # 1. SANKEY DIAGRAMS
    print("1. Generating Sankey Diagrams...")
    
    # Basic Sankey diagram for all agents
    print("   - Basic Sankey diagram for all agents")
    analyzer.sankey_agents(subtitle="All Agents State Transitions")
    
    # Sankey diagram with dwell times
    print("   - Sankey diagram with dwell time information")
    analyzer.sankey_dwelltimes(subtitle="State Transitions with Dwell Times")
    
    # Sankey diagrams by age groups
    print("   - Sankey diagrams by age groups")
    analyzer.sankey_agents_by_age_subplots(bins=[0, 5, 15, 30, 50, 200], scenario="Age-stratified Analysis")
    
    # Sankey diagrams with even age ranges
    print("   - Sankey diagrams with even age ranges")
    analyzer.sankey_agents_even_age_ranges(number_of_plots=3, scenario="Even Age Distribution")
    
    # 2. NETWORK GRAPHS
    print("\n2. Generating Network Graphs...")
    
    # Basic state transition graph
    print("   - Basic state transition graph")
    analyzer.graph_state_transitions(subtitle="State Transition Network", colormap='viridis')
    
    # Curved state transition graph
    print("   - Curved state transition graph with edge thickness")
    analyzer.graph_state_transitions_curved(subtitle="Curved State Transitions", colormap='plasma')
    
    # 3. HISTOGRAMS AND DISTRIBUTIONS
    print("\n3. Generating Histograms and Distributions...")
    
    # Histogram with KDE
    print("   - Histogram with KDE for dwell time distributions")
    analyzer.histogram_with_kde(subtitle="Dwell Time Distribution Analysis")
    
    # Dwell time validation plot
    print("   - Dwell time validation plot")
    analyzer.plot_dwell_time_validation()
    
    # Interactive dwell time validation
    print("   - Interactive dwell time validation")
    analyzer.plot_dwell_time_validation_interactive()
    
    # 4. INTERACTIVE BAR CHARTS
    print("\n4. Generating Interactive Bar Charts...")
    
    # Interactive state transitions bar chart
    print("   - Interactive state transitions bar chart")
    analyzer.barchar_all_state_transitions_interactive(
        dwell_time_bins=[0, 30, 90, 180, 365, float('inf')],
        filter_states=['-1.0.None', '0.0.Latent Slow', '1.0.Latent Fast', '2.0.Active Presymp']
    )
    
    # Interactive reinfections by age
    print("   - Interactive reinfections by age")
    analyzer.reinfections_age_bins_bars_interactive(
        target_states=[0.0, 1.0],  # Active TB states
        barmode='group',
        scenario="Age-stratified Reinfection Analysis"
    )
    
    # Interactive reinfections percentages
    print("   - Interactive reinfections percentages")
    analyzer.reinfections_percents_bars_interactive(
        target_states=[0.0, 1.0],
        scenario="Population Reinfection Analysis"
    )
    
    # Interactive reinfections by state transitions
    print("   - Interactive reinfections by state transitions")
    analyzer.reinfections_bystates_bars_interactive(
        target_states=[0.0, 1.0],
        scenario="State Transition Reinfection Analysis",
        barmode='group'
    )
    
    # 5. STACKED BAR CHARTS
    print("\n5. Generating Stacked Bar Charts...")
    
    # Static stacked bars per agent
    print("   - Static stacked bars showing cumulative time per agent")
    analyzer.stacked_bars_states_per_agent_static()
    
    # Interactive stacked bars by dwell time
    print("   - Interactive stacked bars by dwell time")
    analyzer.stackedbars_dwelltime_state_interactive(bin_size=5, num_bins=15)
    
    # Subplot stacked bars for state transitions
    print("   - Subplot stacked bars for state transitions")
    analyzer.stackedbars_subplots_state_transitions(bin_size=2, num_bins=25)
    
    # 6. CUSTOM TRANSITION ANALYSIS
    print("\n6. Generating Custom Transition Analysis...")
    
    # Custom transition subplots
    print("   - Custom transition subplots")
    custom_transitions = {
        '-1.0.None': ['0.0.Latent Slow', '1.0.Latent Fast'],
        '0.0.Latent Slow': ['2.0.Active Presymp', '-1.0.None'],
        '1.0.Latent Fast': ['2.0.Active Presymp', '-1.0.None']
    }
    analyzer.subplot_custom_transitions(transitions_dict=custom_transitions)
    
    # 7. SURVIVAL ANALYSIS
    print("\n7. Generating Survival Analysis...")
    
    # Kaplan-Meier survival curve
    print("   - Kaplan-Meier survival curve")
    analyzer.plot_kaplan_meier(dwell_time_col='dwell_time')
    
    # 8. USING DWT PLOTTER DIRECTLY
    print("\n8. Using DwtPlotter directly from generated file...")
    
    # Get the file path from the analyzer
    file_path = analyzer.file_path
    print(f"   - Generated data file: {file_path}")
    
    # Create a plotter instance
    plotter = mtb.DwtPlotter(file_path=file_path)
    
    # Demonstrate some additional plots with the plotter
    print("   - Additional histogram with KDE from file")
    plotter.histogram_with_kde(subtitle="From Generated File")
    
    print("   - Additional Sankey diagram from file")
    plotter.sankey_agents(subtitle="From Generated File")
    
    print("\n=== ALL PLOTS COMPLETED ===")
    print(f"Data file saved to: {file_path}")
    
    return analyzer, plotter

def demonstrate_post_processor():
    """Demonstrate the DwtPostProcessor for multiple simulation results"""
    print("\n=== POST PROCESSOR DEMONSTRATION ===")
    print("Note: This would typically be used with multiple simulation results")
    print("from different scenarios stored in a directory.")
    
    # Example of how to use post processor (commented out as we don't have multiple files)
    """
    # Example usage with multiple simulation results
    postproc = mtb.DwtPostProcessor(
        directory='results', 
        prefix='Baseline'
    )
    
    # Generate aggregated plots
    postproc.sankey_agents(subtitle="Aggregated Results")
    postproc.histogram_with_kde(subtitle="Aggregated Distributions")
    postproc.reinfections_percents_bars_interactive(
        target_states=[0.0, 1.0], 
        scenario="Aggregated Reinfection Analysis"
    )
    """

if __name__ == '__main__':
    # Run the comprehensive demonstration
    analyzer, plotter = demonstrate_all_analyzer_plots()
    
    # Demonstrate post processor usage (informational)
    demonstrate_post_processor()
    
    print("\n=== SCRIPT COMPLETED ===")
    print("All plotting methods from the DwtAnalyzer class have been demonstrated.")
    print("Check the generated plots and the data file for detailed analysis.") 