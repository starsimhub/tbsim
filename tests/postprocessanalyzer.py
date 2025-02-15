import tbsim as mtb

directory = '/Users/Mine/TEMP/results' 
other = '-0206'
scenarios_13 = [   
                'HighDecliningLSHTM',
                'HighDecliningTBsim',
                'HighFlatLSHTM',
                'HighFlatTBsim',
                'LowDecliningLSHTM',
                'LowDecliningTBsim',
                'LowFlatLSHTM',
                'LowFlatTBsim'
             ]
scenarios_feb06 = [
               'ACTLSHTM',
                'ACTTBsim',
                'archive',
                'BaselineLSHTM',
                'BaselineTBsim',
                'HighTransmissionLSHTM',
                'HighTransmissionTBsim',
                'MissedSubclinicalLSHTM',
                'MissedSubclinicalTBSim',   
                ]

for sce in scenarios_13:
    an = mtb.DwtPostProcessor(directory= directory, prefix=f"{sce}{other}")
    # an.sankey_agents(subtitle=sce)
    # an.graph_state_transitions_curved(subtitle=sce, graphseed=39)
    an.reinfections_percents_bars_interactive(target_states=[0.0, -1], scenario=sce)
    an.reinfections_age_bins_bars_interactive(target_states=[0.0, -1], scenario=sce)
    an.reinfections_bystates_bars_interactive(target_states=[0.0, -1], scenario=sce) 


"""
    Sample of using plotter alone once the csv files are generated
"""
# file = '/Users/mine/git/tb_acf/results/results/BaselineTBsim34935.csv'
# plotter = mtb.DwtPlotter(file)
# plotter.sankey_agents()

# /Users/mine/TEMP/results/ALL_HighDecliningLSHTM-0206.csv
