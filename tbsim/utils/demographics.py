import pandas as pd
import numpy as np
import os

def load_age_data(source='default', file_path=''):
    """
    Load population data from a CSV file or use default data.
    """
    if source == 'default':
        # Default population data
        # Gathered from WPP, https://population.un.org/wpp/Download/Standard/MostUsed/
        age_data = pd.DataFrame({ 
            'age': np.arange(0, 101, 5),
            'value': [5791, 4446, 3130, 2361, 2279, 2375, 2032, 1896, 1635, 1547, 1309, 1234, 927, 693, 460, 258, 116, 36, 5, 1, 0]  # 1960
        })
    elif source == 'json':
        if not file_path:
            raise ValueError("file_path must be provided when source is 'json'.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        data = pd.read_json(file_path)
        age_data = pd.DataFrame(data)
    else:
        raise ValueError("Invalid source. Use 'default' or 'json'.")
    return age_data
