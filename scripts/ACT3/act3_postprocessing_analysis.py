import tbsim as mtb
import itertools
import datetime as dt


"""
The directory below is where the data files (*.CSV) are saved when running the 
simulations with the ** DwtAnalyzer **. 
NOTE: The analyzer will create the csv files with the data needed to generate the charts.
The scenarios are the names of the csv files without the prefix and the postfix
PREFIX: usually the date of the simulation

OPTION 1: Get the scenarios from the files in the directory (better for automation - a one time execution)
```python
import os
directory = '/Users/mine/git/tb_acf/tb_acf/results/'
files = os.listdir(directory)
scenarios = [f.split('-')[0] for f in files if f.endswith('.csv') and not 'WithReinfection' in f]
scenarios = [f.replace('.csv', '') for f in scenarios]
scenarios = list(set(scenarios))
scenarios.sort()
print(scenarios)
```
# OPTION 2: Define the scenarios manually <-- better for clarity on what is actually being processed
"""

debug = 1 # 1 = invidual_charts, 2= percentaje_of_reinfections


directory = '/Users/mine/git/tb_acf/tb_acf/results/' 

# (OPTION 2): Define the scenarios manually
scenarios = [   
            'HighDecliningLSHTM',
            'HighDecliningLSHTMAcute',
            'HighDecliningTBsim',
            'HighFlatLSHTM',
            'HighFlatLSHTMAcute',
            'HighFlatTBsim',
            'LowDecliningLSHTM',
            'LowDecliningLSHTMAcute',
            'LowDecliningTBsim',
            'LowFlatLSHTM',
            'LowFlatLSHTMAcute',
            'LowFlatTBsim'
            ]


def individual_charts():
  other = '-0213'
  for sce in scenarios:
      an = mtb.DwtPostProcessor(directory= directory, prefix=f"{sce}{other}")
      # an.reinfections_age_bins_bars_interactive(target_states=[-1.0, 0.0, 1.0]  ,  scenario=sce)
      # an.barchar_all_state_transitions_interactive()
      # an.graph_state_transitions(subtitle=sce)
      an.sankey_agents_by_age_subplots(bins=[5, 200], scenario=sce)


def percentaje_of_reinfections():
  import matplotlib.pyplot as plt
  import pandas as pd
  postfix = '_WithReinfection.csv'
  # use subplots to chart all the percentaje values in one chart
  fig, ax = plt.subplots(figsize=(15, 10))
  all_scenarios = pd.DataFrame()

  for sce in scenarios:
    infection_states = [-1.0, 0.0, 1.0]
    filename = f"{directory}/{sce}{postfix}"
    df = pd.read_csv(filename)
    df = df[df['going_to_state_id'].isin(infection_states)]
    df['scenario'] = sce
    data = df[['scenario', 'infection_num', 'percent']].groupby(['scenario', 'infection_num']).sum().reset_index()
    all_scenarios = pd.concat([all_scenarios, data])

  data = all_scenarios.groupby(['scenario', 'infection_num']).sum().reset_index()
  print(data)
  markers = itertools.cycle(['o', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd', '1', '2'])
  for scenario in data['scenario'].unique():
    scenario_data = data[data['scenario'] == scenario]
    ax.plot(scenario_data['infection_num'], scenario_data['percent'], label=scenario, marker=markers.__next__())
  ax.set_xticks(range(0, 40))
  ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
  
  ax.set_xlabel('Infection Number')
  ax.set_ylabel('Percent')
  ax.set_title(f'Percentage of Reinfections by Scenario \n {dt.datetime.now().strftime("%Y-%m-%d")} \n Data from Feb. 13')
  ax.legend()
  plt.show()
  

def to_filename_friendly(string=''):
  import re


  string = "".join([c if c.isalpha() else "_" for c in string])
  return re.sub(r'[^a-zA-Z0-9]', '', string)

if __name__ == '__main__':
  if debug == 1:
    individual_charts()
  if debug == 2:
    percentaje_of_reinfections()