import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tbsim
import sciris as sc
import tbsim.config as cfg

st = tbsim.nutritionenums.eBmiStatus
desc = tbsim.nutritionenums.descriptions

Adults_Control_Arm = {
    st.SEVERE_THINNESS.name: [130, 14, 4, 0, 0],
    st.MODERATE_THINNESS.name: [66, 91, 23, 0, 0],
    st.MILD_THINNESS.name: [14, 93, 280, 34, 0],
    st.NORMAL_WEIGHT.name: [1, 8, 138, 1611, 6],
    st.OVERWEIGHT.name: [0, 0, 0, 74, 237]
}

Adults_Intervention_Arm = {
    st.SEVERE_THINNESS.name: [127, 5, 0, 0, 0],
    st.MODERATE_THINNESS.name: [85, 96, 18, 0, 0],
    st.MILD_THINNESS.name: [40, 177, 310, 34, 0],
    st.NORMAL_WEIGHT.name: [4, 31, 341, 1611, 6],
    st.OVERWEIGHT.name: [0, 0, 0, 74, 237]
}

index= [desc.SEVERE_THINNESS[0], 
        desc.MODERATE_THINNESS[0], 
        desc.MILD_THINNESS[0], 
        desc.NORMAL_WEIGHT[0], 
        desc.OVERWEIGHT[0]]

def get_variable_name(var):
    globals_dict = globals()
    var_name = [name for name in globals_dict if globals_dict[name] is var]
    if len(var_name) == 0:
        return None
    return var_name[0].replace("_", " ")
        
def exclude_diagonal(df):
    """
    Set diagonal values to a placeholder value to exclude them from the heatmap.
    Args:
        df (pd.DataFrame): DataFrame with data.
    Returns:
        pd.DataFrame: DataFrame with diagonal values set to a placeholder value.
    """
    df_with_placeholder = df.copy()
    np.fill_diagonal(df_with_placeholder.values, -1)  # Set diagonal to a placeholder value (-1)
    return df_with_placeholder

def exclude_below_diagonal(df):
    """
    Set values below the diagonal to a placeholder value to exclude them from the heatmap.
    Args:
        df (pd.DataFrame): DataFrame with data.
    Returns:
        pd.DataFrame: DataFrame with values below the diagonal set to a placeholder value.
    """
    df_with_placeholder = df.copy()
    num_rows, num_cols = df.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if i > j:  # Below diagonal
                df_with_placeholder.iat[i, j] = -1  # Set to placeholder value
    return df_with_placeholder

def convert_to_percentages(df):
    """
    Convert values in the DataFrame to percentages of the total sum.
    Args:
        df (pd.DataFrame): DataFrame with data.
    Returns:
        pd.DataFrame: DataFrame with percentages.
    """
    df_sum = df.sum().sum()
    if df_sum == 0:
        return df  # Return unchanged if the sum is zero to avoid division by zero
    df_percentage = (df / df_sum * 100).round(2)  # Calculate percentages
    return df_percentage

def plot_heatmap(index={}, data=[{}], ex_diagonal=False, ex_below_diagonal=False, use_percentages= False, colormap='OrRd', filename='heatmap'):
    """
    Plot heatmap data in subplots, excluding diagonal values.
    Args:
        index (list): The index labels for the DataFrame.
        data (list): A list of dictionaries containing the data to plot.
    """
    # Creating DataFrames
    dfs = [pd.DataFrame(d, index=index) for d in data]
    tots = [df.sum().sum() for df in dfs]
    if use_percentages:
        dfs = [convert_to_percentages(df) for df in dfs]  # Convert to percentages
    if ex_diagonal: 
        dfs = [exclude_diagonal(df) for df in dfs]  # Exclude diagonal values
    if ex_below_diagonal:
        dfs = [exclude_below_diagonal(df) for df in dfs]  # Exclude values below diagonal
    # Creating a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    titles = [get_variable_name(data[0]), get_variable_name(data[1])]

    # Define a custom colormap that makes the placeholder value less visible
   
    cmap = plt.get_cmap(colormap)
    cmap.set_under('white')  # Set color for values below the minimum of the colormap

    for ax, df, title, tot in zip(axes, dfs, titles, tots):
        # Plotting the heatmap
        value_type = '(%)' if use_percentages else ''
        heatmap = ax.imshow(df, cmap=cmap, interpolation='nearest', vmin=0.1)  # Use vmin to exclude placeholder color
        ax.set_title(title, pad=20)
        ax.set_xlabel(f'{tot} people - BMI after 6 months {value_type}')
        ax.set_ylabel(f'Baseline BMI (kg/m²) in the {title}')
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                if df.iloc[i, j] != -1:  # Only display text for non-placeholder values
                    ax.text(j, i, df.iloc[i, j], ha='center', va='center', color='black')
        fig.colorbar(heatmap, ax=ax)
        ax.xaxis.set_ticks_position('top')

    plt.tight_layout(pad=4.0)
    fig.suptitle(f"BMI at baseline and at the end of 6 months in adult household contacts\n{filename.replace('_', ' ')}", fontsize=14)
    sc.savefig(f"{filename}_{cfg.FILE_POSTFIX}.png", folder=cfg.RESULTS_DIRECTORY)
    print(f"Open plot: '{cfg.RESULTS_DIRECTORY}\{filename}_{cfg.FILE_POSTFIX}.png'")
    plt.close()
    
    

if __name__ == '__main__':
    plot_heatmap(index, [Adults_Control_Arm, Adults_Intervention_Arm], False, False, False, 'OrRd', filename="All_Values" )
    plot_heatmap(index, [Adults_Control_Arm, Adults_Intervention_Arm], True, False, True, 'YlGnBu', filename="BMI_With_Changes" )
    plot_heatmap(index, [Adults_Control_Arm, Adults_Intervention_Arm], True, True, True, 'BuGn', filename="BMI_Improvements_Only" )
    
    print(f"Figures: {cfg.RESULTS_DIRECTORY}")
    