import matplotlib.pyplot as plt
import pandas as pd

# Data from the image
data = {
    "SEVERE   <16 ": [130, 66, 14, 1, 0],
    "MODERATE 16.0-16.99 ": [14, 91, 93, 8, 0],
    "MILD     17.0-18.49 ": [4, 23, 280, 138, 0],
    "NORMAL   18.5-24.99 ": [0, 0, 34, 1611, 74],
    "OVER     ≥25.0 ": [0, 0, 0, 6, 237]
}

index = ["SEVERE   <16 ", 
         "MODERATE 16.0-16.99 ", 
         "MILD     17.0-18.49 ", 
         "NORMAL   18.5-24.99 ", 
         "OVER     ≥25.0 "]

# Creating a DataFrame
df = pd.DataFrame(data, index=index)

# Plotting the chart
plt.figure(figsize=(10, 8))
heatmap = plt.imshow(df, cmap='ocean_r', interpolation='nearest')

# Adding color bar
plt.colorbar(heatmap)

# Adding labels
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.yticks(range(len(df.index)), df.index)
plt.xlabel('BMI after 6 months')
plt.ylabel('Baseline BMI (kg/m²)')

# Adding title
plt.title('BMI Transition Over 6 Months')

# Annotating the cells with the respective counts
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        plt.text(j, i, df.iloc[i, j], ha='center', va='center', color='black')

# Show plot
plt.tight_layout()
plt.show()