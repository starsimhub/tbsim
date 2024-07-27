import matplotlib.pyplot as plt
import pandas as pd

# Data from the image
data = {
    "<16 kg/m²": [130, 66, 14, 1, 0],
    "16.0-16.99 kg/m²": [14, 91, 93, 8, 0],
    "17.0-18.49 kg/m²": [4, 23, 280, 138, 0],
    "18.5-24.99 kg/m²": [0, 0, 34, 1611, 74],
    "≥25.0 kg/m²": [0, 0, 0, 6, 237]
}

index = ["<16 kg/m²", "16.0-16.99 kg/m²", "17.0-18.49 kg/m²", "18.5-24.99 kg/m²", "≥25.0 kg/m²"]

# Creating a DataFrame
df = pd.DataFrame(data, index=index)

# Plotting the chart
plt.figure(figsize=(10, 8))
heatmap = plt.imshow(df, cmap='coolwarm', interpolation='nearest')

# Adding color bar
plt.colorbar(heatmap)

# Adding labels
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.yticks(range(len(df.index)), df.index)
plt.xlabel('BMI after 6 months')
plt.ylabel('Baseline BMI')

# Adding title
plt.title('BMI Transition Over 6 Months')

# Annotating the cells with the respective counts
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        plt.text(j, i, df.iloc[i, j], ha='center', va='center', color='black')

# Show plot
plt.tight_layout()
plt.show()