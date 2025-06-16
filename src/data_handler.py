import pandas as pd
from pathlib import Path

class DataHandler:
    """Handles loading and validation of historical data from a CSV file."""
    
    def __init__(self, file_path, required_columns):
        self.file_path = Path(file_path)
        self.required_columns = required_columns
        self.df = self._load_data()

    def _load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        if not self.required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain these columns: {self.required_columns}")
        return df

    def get_head(self, n=4):
        return self.df[list(self.required_columns)].head(n)