import os
import numpy as np
import pandas as pd
from tbsim import TBS
from matplotlib import colormaps as cm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.dates as mdates

differences_only = True

ctrl = pd.read_csv('/Users/dklein/GIT/tb_acf/tb_acf/results/Scen_2x2_DEBUG/High+Flat (TBsim)/transitions/Control_813747_state_transitions.csv')
intv = pd.read_csv('/Users/dklein/GIT/tb_acf/tb_acf/results/Scen_2x2_DEBUG/High+Flat (TBsim)/transitions/Intervention_813747_state_transitions.csv')

# TBS is an Enum, look up the name of the integer state from_state in the enum
def get_state_name(from_state):
    try:
        return TBS(int(float(from_state))).name
    except ValueError:
        return from_state

ctrl['from_state'] = ctrl['from_state'].apply(get_state_name)
ctrl['to_state'] = ctrl['to_state'].apply(get_state_name)
ctrl['label'] = 'Control'


intv['from_state'] = intv['from_state'].apply(get_state_name)
intv['to_state'] = intv['to_state'].apply(get_state_name)
intv['label'] = 'Intervention'

df = pd.concat([ctrl, intv]) \
    .set_index(['uid', 'label']) \
    .sort_index() \
    [['t_enter', 't', 'from_state', 'to_state', 'dwell']]

df['t'] = mdates.date2num(pd.to_datetime(df['t']))
df['t_enter'] = mdates.date2num(pd.to_datetime(df['t_enter']))

fig, ax = plt.subplots(1, 1, figsize=(16, 10))

cmap = cm['plasma']
states = list(set(df['from_state'].unique()) | set(df['to_state'].unique()))
#colors = {key:color for key ,color in zip(states, cmap(np.linspace(0, 1, len(states))))}

colors = {
    'BORN': np.array([0.4, 0.4, 0.4, 1]),
    'INIT':np.array([0.6, 0.6, 0.6, 1]),
    'NONE': np.array([0, 1, 0, 1]),
    'ACTIVE_PRESYMP': np.array([250, 163, 2, 255])/255,
    'ACTIVE_SMNEG': np.array([168, 50, 103, 255])/255,
    'ACTIVE_SMPOS': np.array([1,0,0,1]),
    'ACTIVE_EXPTB': np.array([255, 192, 203, 255])/255,
    'LATENT_SLOW': np.array([7, 189, 245, 255])/255,
    'LATENT_FAST': np.array([5, 60, 242, 255])/255,
    'DEAD': np.array([0,0,0,1]),
    'END': np.array([0.2,0.2,0.2,1]),
}

idx = 0
for _, (uid, dat_uid) in enumerate(df.groupby('uid')):

    if differences_only:
        # Skip if identical /// well if sum of t_enter is the same as a proxy
        t_enter_sum = dat_uid['t_enter'].groupby('label').sum()
        if len(t_enter_sum)==2 and t_enter_sum.loc['Control'] == t_enter_sum.loc['Intervention']:
            continue

    for jdx, (lbl, dat) in enumerate(dat_uid.groupby('label')):
        jj = 0 if lbl == 'Control' else 1
        y = 3*idx+jj

        col_from = dat['from_state'].apply(lambda x: colors.get(x, 'black'))
        patches = [Rectangle((t_enter, y), width=t-t_enter, height=0.8, facecolor=c) for (t_enter, t, c) in zip(dat['t_enter'].values, dat['t'].values, col_from)]
        pc = PatchCollection(patches=patches, match_original=True)
        ax.add_collection(pc)

        col_to = dat['to_state'].apply(lambda x: colors.get(x, 'black'))

        lines = [np.row_stack([(t_enter, y-0.1), (t_enter, y+0.9)]) for t_enter in dat['t_enter'].values] \
            + [np.row_stack([(t, y-0.1), (t, y+0.9)]) for t in dat['t'].values]
        lc = LineCollection(lines, colors=col_from.values.tolist() + col_to.values.tolist(), linewidths=1)
        ax.add_collection(lc)
        #ax.scatter(dat['t_enter'], [y]*len(dat), color=col_from, s=1, marker='|')
        #ax.scatter(dat['t'], [y]*len(dat), color=col_to, s=1, marker='|')
    idx += 1

for lbl, col in colors.items():
    ax.scatter(0, 0, color=col, label=lbl)

ax.axvline(mdates.date2num(pd.to_datetime('2014-06-01')), color='k', linestyle='--', linewidth=0.5)
ax.axvline(mdates.date2num(pd.to_datetime('2015-06-01')), color='k', linestyle='--', linewidth=0.5)
ax.axvline(mdates.date2num(pd.to_datetime('2016-06-01')), color='k', linestyle='--', linewidth=0.5)

ax.autoscale_view()
ax.legend()

# Format the x-axis to show dates
date_format = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)

# Ensure proper date alignment
fig.autofmt_xdate()

print('saving...')
fig.savefig('individuals.pdf')
plt.show()
print('done')
