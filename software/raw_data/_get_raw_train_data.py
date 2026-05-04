import os
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent.parent / 'data'

def get_raw_train_data(layer_index: int, data_folder_path=None) -> pd.DataFrame:
    if data_folder_path is None:
        data_folder_path = _DATA_DIR
    if layer_index >= 24:
        print(f"Błąd: layer_index {layer_index} jest poza zakresem (dostępnych warstw: 24).")
        return pd.DataFrame()

    path = os.path.join(data_folder_path, f'activations/raw/train/layer_{layer_index:02d}.parquet')
    if not os.path.exists(path):
        print(f"Nie znaleziono pliku: {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)
