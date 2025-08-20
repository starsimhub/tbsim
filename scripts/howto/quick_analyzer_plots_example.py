import tbsim as mtb
import starsim as ss
import starsim_examples as sse
import sciris as sc 

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

    dwell_analyzer = mtb.DwtAnalyzer(adjust_to_unit=True  # TODO: Check if adjust_to_unit is still needed in v3, dt=1.0, scenario_name='quick_plots_example')

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        pars=sim_params,
        analyzers=dwell_analyzer,
    )
    sim.pars.verbose = 30/365

    return sim

def demonstrate_essential_plots():
    """Demonstrate the most essential plotting methods from the DwtAnalyzer class"""
    
    print("Building and running TB simulation...")
    sim_tb = build_tbsim()
    sim_tb.run()
    
    # Extract the analyzer
    analyzer = sim_tb.analyzers[0]
    
    print("\n=== ESSENTIAL DWT ANALYZER PLOTS ===\n")
    
    # 1. SANKEY DIAGRAM - Most important for understanding state transitions
    print("1. Generating Sankey Diagram (shows agent flow between states)...")
    analyzer.sankey_agents(subtitle="TB State Transitions")
    
    # 2. NETWORK GRAPH - Shows state transition network with statistics
    print("2. Generating Network Graph (shows transition network with statistics)...")
    analyzer.graph_state_transitions_curved(subtitle="State Transition Network")
    
    # 3. HISTOGRAM WITH KDE - Shows dwell time distributions
    print("3. Generating Histogram with KDE (shows dwell time distributions)...")
    analyzer.histogram_with_kde(subtitle="Dwell Time Analysis")
    
    # 4. INTERACTIVE STATE TRANSITIONS - Detailed interactive analysis
    print("4. Generating Interactive State Transitions Chart...")
    analyzer.barchar_all_state_transitions_interactive(
        dwell_time_bins=[0, 30, 90, 180, 365, float('inf')]
    )
    
    # 5. REINFECTION ANALYSIS - Important for TB epidemiology
    print("5. Generating Reinfection Analysis...")
    analyzer.reinfections_percents_bars_interactive(
        target_states=[0.0, 1.0],  # Active TB states
        scenario="Reinfection Analysis"
    )
    
    # 6. AGE-STRATIFIED ANALYSIS - Shows age-specific patterns
    print("6. Generating Age-stratified Sankey Diagrams...")
    analyzer.sankey_agents_by_age_subplots(
        bins=[0, 5, 15, 30, 50, 200], 
        scenario="Age-stratified Analysis"
    )
    
    print("\n=== ESSENTIAL PLOTS COMPLETED ===")
    print(f"Data file saved to: {analyzer.file_path}")
    
    return analyzer

def demonstrate_plotter_usage():
    """Show how to use the DwtPlotter with existing data files"""
    print("\n=== USING DWT PLOTTER WITH EXISTING DATA ===")
    
    # This would typically be used with existing data files
    # For demonstration, we'll show the pattern:
    
    """
    # Example usage with existing data file
    plotter = mtb.DwtPlotter(file_path='results/my_simulation.csv')
    
    # Generate the same essential plots
    plotter.sankey_agents(subtitle="From Data File")
    plotter.graph_state_transitions_curved(subtitle="From Data File")
    plotter.histogram_with_kde(subtitle="From Data File")
    """
    
    print("To use DwtPlotter with existing data files:")
    print("1. Create plotter: plotter = mtb.DwtPlotter(file_path='your_file.csv')")
    print("2. Generate plots: plotter.sankey_agents(), plotter.histogram_with_kde(), etc.")
    print("3. All the same plotting methods are available!")

if __name__ == '__main__':
    # Run the essential plots demonstration
    analyzer = demonstrate_essential_plots()
    
    # Show how to use plotter with existing data
    demonstrate_plotter_usage()
    
    print("\n=== QUICK START COMPLETED ===")
    print("These 6 plots provide the most important insights into TB simulation dynamics:")
    print("1. Sankey Diagram - Overall state transition patterns")
    print("2. Network Graph - Detailed transition statistics")
    print("3. Histogram with KDE - Dwell time distributions")
    print("4. Interactive State Transitions - Detailed interactive analysis")
    print("5. Reinfection Analysis - TB-specific epidemiology")
    print("6. Age-stratified Analysis - Age-specific patterns")
    print("\nFor more advanced plots, see: comprehensive_analyzer_plots_example.py") 