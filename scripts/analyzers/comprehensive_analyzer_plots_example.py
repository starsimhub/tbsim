#!/usr/bin/env python3
"""
Comprehensive Analyzer Plots Example

This script demonstrates all available plotting methods in the `DwtAnalyzer` class from the `tbsim` package.
It is adapted from the original Jupyter notebook for interactive exploration.

Usage:
    python comprehensive_analyzer_plots_example.py
"""

import tbsim as mtb
import starsim as ss
import pandas as pd

TBS = mtb.TBS

def build_tbsim(sim_pars=None):
    """Build a TB simulation with dwell time analyzer"""
    sim_params = dict(
        start = ss.date('2013-01-01'),
        stop = ss.date('2016-12-31'),
        rand_seed=1,
        dt=ss.days(7),
    )
    
    if sim_pars is not None:
        sim_params.update(sim_pars)
        
    pop = ss.People(n_agents=1000)
    tb_params = dict(
        beta=ss.peryear(0.25),
        init_prev=ss.bernoulli(p=0.35),
        rel_sus_latentslow=0.3,
        dt=ss.days(7),
    )
    
    tb = mtb.TB(tb_params)
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    dwell_analyzer = mtb.DwtAnalyzer( scenario_name='comprehensive_plots_example')
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        pars=sim_params,
        analyzers=dwell_analyzer,
    )
    sim.pars.verbose = 0
    return sim

def run_simulation():
    """Run the simulation and extract the analyzer"""
    print("Building and running TB simulation...")
    sim_tb = build_tbsim()
    sim_tb.run()
    analyzer = sim_tb.analyzers[0]
    
    # Load the data from the saved file since the analyzer doesn't automatically load it
    if hasattr(analyzer, 'file_path') and analyzer.file_path:
        print(f"Loading data from: {analyzer.file_path}")
        analyzer.data = analyzer.__cleandata__(analyzer.file_path)
        print(f"Loaded {len(analyzer.data)} records")
    else:
        print("Warning: No data file found for analyzer")
    
    return analyzer

def demonstrate_sankey_plots(analyzer):
    """Demonstrate Sankey diagrams for state transitions"""
    print("\n" + "="*60 + "\n" + "1. SANKEY DIAGRAMS" + "\n" + "="*60)
    
    # Basic Sankey diagram for all agents
    print("1.1 Basic Sankey diagram for all agents")
    analyzer.sankey_agents()
    
    # Sankey diagram with dwell times
    print("1.2 Sankey diagram with dwell times")
    analyzer.sankey_dwelltimes(subtitle="State Transitions with Dwell Times")
    
    # Sankey diagrams by age groups
    print("1.3 Sankey diagrams by age groups")
    analyzer.sankey_agents_by_age_subplots(bins=[0, 5, 15, 30, 50, 200], scenario="Age-stratified Analysis")
    
    # Sankey diagrams with even age ranges
    print("1.4 Sankey diagrams with even age ranges")
    analyzer.sankey_agents_even_age_ranges(number_of_plots=3, scenario="Even Age Distribution")

def demonstrate_network_graphs(analyzer):
    """Visualize state transition networks"""
    print("\n" + "="*60 + "\n" + "2. NETWORK GRAPHS" + "\n" + "="*60)
    
    print("2.1 Basic state transition network")
    analyzer.graph_state_transitions(subtitle="State Transition Network", colormap='tab20')
    
    
    print("2.2 Curved state transitions")
    analyzer.graph_state_transitions_curved(subtitle="Curved State Transitions", colormap='plasma')
    

def demonstrate_histograms(analyzer):
    """Explore dwell time distributions"""
    print("\n" + "="*60 + "\n" + "3. HISTOGRAMS AND DISTRIBUTIONS" + "\n" + "="*60)
    
    print("3.1 Histogram with KDE")
    analyzer.histogram_with_kde()
    
    
    print("3.2 Dwell time validation")
    analyzer.plot_dwell_time_validation()
    
    
    print("3.3 Interactive dwell time validation")
    analyzer.plot_dwell_time_validation_interactive()
    

def demonstrate_interactive_bar_charts(analyzer):
    """Interactive bar charts for state transitions and reinfections"""
    print("\n" + "="*60 + "\n" + "4. INTERACTIVE BAR CHARTS" + "\n" + "="*60)
    
    print("4.1 Interactive bar chart for state transitions")
    analyzer.barchar_all_state_transitions_interactive(
        dwell_time_bins=[0, 5, 15, 30, 60, 120, float('inf')],  # More appropriate bins for the data
        filter_states=['-1.0.None', '0.0.Latent Slow', '1.0.Latent Fast', '2.0.Active Presymp']  # Correct state name format
    )
    
    
    print("4.2 Age-stratified reinfection analysis")
    analyzer.reinfections_age_bins_bars_interactive(
        target_states=[0.0, 1.0],
        barmode='group',
        scenario="Age-stratified Reinfection Analysis"
    )
    

    
    print("4.3 Population reinfection analysis")
    analyzer.reinfections_percents_bars_interactive(
        target_states=[0.0, 1.0],
        scenario="Population Reinfection Analysis"
    )
    
    
    print("4.4 State transition reinfection analysis")
    analyzer.reinfections_bystates_bars_interactive(
        target_states=[0.0, 1.0],
        scenario="State Transition Reinfection Analysis",
        barmode='group'
    )
    

def demonstrate_stacked_bar_charts(analyzer):
    """Stacked bar charts for cumulative time and dwell time analysis"""
    print("\n" + "="*60 + "\n" + "5. STACKED BAR CHARTS" + "\n" + "="*60)
    
    print("5.1 Stacked bars per agent")
    analyzer.stacked_bars_states_per_agent_static()
    

    print("5.2 Interactive stacked bars by dwell time")
    analyzer.stackedbars_dwelltime_state_interactive(bin_size=5, num_bins=15)
    

    print("5.3 Subplot stacked bars")
    analyzer.stackedbars_subplots_state_transitions(bin_size=2, num_bins=25)
    

def demonstrate_custom_transitions(analyzer):
    """Custom transition subplots"""
    print("\n" + "="*60 + "\n" + "6. CUSTOM TRANSITION ANALYSIS" + "\n" + "="*60)    
    print("6.1 Custom transition subplots")
    custom_transitions = {
        '-1.0.None': ['0.0.Latent Slow', '1.0.Latent Fast'],
        '0.0.Latent Slow': ['2.0.Active Presymp', '-1.0.None'],
        '1.0.Latent Fast': ['2.0.Active Presymp', '-1.0.None']
    }
    analyzer.subplot_custom_transitions(transitions_dict=custom_transitions)
    

def demonstrate_survival_analysis(analyzer):
    """Kaplan-Meier survival curve for dwell times"""
    print("\n" + "="*60 + "\n" + "7. SURVIVAL ANALYSIS" + "\n" + "="*60)
    
    print("7.1 Kaplan-Meier survival curve")
    analyzer.plot_kaplan_meier(dwell_time_col='dwell_time')
    

def demonstrate_dwtplotter_directly(analyzer):
    """Demonstrate additional plots using the DwtPlotter class"""
    print("\n" + "="*60 + "\n" + "8. USING DWTPLOTTER DIRECTLY" + "\n" + "="*60)
    print("8. USING DWTPLOTTER DIRECTLY")
    
    print("8.1 Using DwtPlotter with generated data file")
    file_path = analyzer.file_path
    print(f'Generated data file: {file_path}')
    plotter = mtb.DwtPlotter(file_path=file_path)
    plotter.histogram_with_kde(subtitle="From Generated File")
    
    plotter.sankey_agents(subtitle="From Generated File")
    

def demonstrate_post_processor():
    """Example usage of the DwtPostProcessor for multiple simulation results"""
    print("\n" + "="*60 + "\n" + "9. POST PROCESSOR DEMONSTRATION" + "\n" + "="*60)
    print("9. POST PROCESSOR DEMONSTRATION")
    
    print("9.1 Post processor example (informational)")
    print("Example usage (commented out):")
    print("# postproc = mtb.DwtPostProcessor(directory='results', prefix='Baseline')")
    print("# postproc.sankey_agents(subtitle=\"Aggregated Results\")")
    print("# postproc.histogram_with_kde(subtitle=\"Aggregated Distributions\")")
    print("# postproc.reinfections_percents_bars_interactive(")
    print("#     target_states=[0.0, 1.0],")
    print("#     scenario=\"Aggregated Reinfection Analysis\"")
    print("# )")

def main():
    """Main function to demonstrate all plotting capabilities"""
    print("TB Simulation Analyzer - Comprehensive Plots Example")
    print("=" * 60 + "\n" + "This script demonstrates all available plotting methods in the DwtAnalyzer class." + "\n" + "=" * 60 )
    print("This script demonstrates all available plotting methods in the DwtAnalyzer class.")
    print("All plots will be displayed interactively.")
    print()
    
    try:
        # Run the simulation
        analyzer = run_simulation()
        
        # Demonstrate each category of plots
        demonstrate_sankey_plots(analyzer)
        demonstrate_network_graphs(analyzer)
        demonstrate_histograms(analyzer)
        demonstrate_interactive_bar_charts(analyzer)
        demonstrate_stacked_bar_charts(analyzer)
        demonstrate_custom_transitions(analyzer)
        demonstrate_survival_analysis(analyzer)
        demonstrate_dwtplotter_directly(analyzer)
        demonstrate_post_processor()
        
        print("\n" + "=" * 60 + "\n" + "🎉 COMPREHENSIVE PLOTS DEMONSTRATION COMPLETED!" + "\n" + "=" * 60)
        print("=" * 60 + "\n" + "All plotting methods have been demonstrated." + "\n" + "=" * 60 + "\n")
        print("Check the generated data file for further analysis:")
        print(f"Data file: {analyzer.file_path}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
