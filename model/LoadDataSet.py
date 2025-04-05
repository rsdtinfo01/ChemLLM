import os

import pandas as pd
from IPython.display import display, HTML


# Load ChemLLM datasets
def load_data():
    # Dynamically get the path of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = base_dir+"TBD"
    print(file_path)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')  # Center-align column headers
    chem_data = pd.read_csv(file_path, encoding='latin-1', names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])
    print("sample Data:")
    display(HTML(chem_data.head().to_html()))
    return chem_data.sample(100);
