import matplotlib.pyplot as plt
import pandas as pd

# Data from the image
Adults_Control_Arm = {
    "<16 kg/m2": [130, 14, 4, 0, 0],
    ">16-0-16-99 kg/m2": [66, 91, 23, 0, 0],
    ">17-0-18-49 kg/m2": [14, 93, 280, 34, 0],
    ">18-5-24-99 kg/m2": [1, 8, 138, 1611, 6],
    ">25-0 kg/m2": [0, 0, 0, 74, 237]
}

Adults_Intervention_Arm = {
 " <16 kg/m2": [127, 5, 0, 0, 0],
    ">16-0-16-99 kg/m2": [85, 96, 18, 0, 0],
    ">17-0-18-49 kg/m2": [40, 177, 310, 34, 0],
    ">18-5-24-99 kg/m2": [4, 31, 341, 1611, 6],
    ">25-0 kg/m2": [0, 0, 0, 74, 237]
}

index= ["Severe\n <16 ",
        "Moderate\n 16.0-16.99 ", 
        "Mild\n 17.0-18.49 ", 
        "Normal\n 18.5-24.99 ", 
        "Over\n ≥25.0 "]

def get_variable_name(var):
    globals_dict = globals()
    var_name = [name for name in globals_dict if globals_dict[name] is var]
    if len(var_name) == 0:
        return None
    return var_name[0].replace("_", " ")
    

def plot_heatmapdata_sp(index={}, data=[{}]):
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

    plt.tight_layout(pad=4.0)
    fig.suptitle('RATIONS - BMI Transition Over 6 Months', fontsize=16)
    plt.show()    
    
if __name__ == '__main__':
    plot_heatmapdata_sp(index, [Adults_Control_Arm, Adults_Intervention_Arm])