
import matplotlib.pyplot as plt
import pandas as pd
import tbsim

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
    

def plot_heatmap(index={}, data=[{}]):
    """
    Plot heatmap data in subplots.
    Args:
        index (list): The index labels for the DataFrame.
        data (list): A list of dictionaries containing the data to plot.
    """
    # Creating DataFrames
    dfs = [pd.DataFrame(d, index=index) for d in data]
    
    # Creating a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    titles = [get_variable_name(data[0]), get_variable_name(data[1])]

    for ax, df, title in zip(axes, dfs, titles):
        # Plotting the heatmap
        heatmap = ax.imshow(df, cmap='OrRd', interpolation='nearest')
        ax.set_title(title, pad=20)
        ax.set_xlabel('BMI after 6 months')
        ax.set_ylabel(f'Baseline BMI (kg/m²) in the {title}')
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                ax.text(j, i, df.iloc[i, j], ha='center', va='center', color='black')
        fig.colorbar(heatmap, ax=ax)
        ax.xaxis.set_ticks_position('top')

    plt.tight_layout(pad=4.0)
    fig.suptitle('BMI at baseline and at the end of 6 months in adult household contacts', fontsize=14)
    
    plt.show()    
    
if __name__ == '__main__':
    plot_heatmap(index, [Adults_Control_Arm, Adults_Intervention_Arm])