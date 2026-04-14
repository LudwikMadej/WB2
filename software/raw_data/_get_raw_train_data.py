import os
import pandas as pd

def get_raw_train_data(layer_index: int, data_folder_path='../data/') -> pd.DataFrame:
    if layer_index >= 24:
        print(f"Błąd: layer_index {layer_index} jest poza zakresem (dostępnych warstw: 24).")
        return pd.DataFrame()

    path = os.path.join(data_folder_path, f'activations/train/layer_{layer_index:02d}.parquet')
    if not os.path.exists(path):
        print(f"Nie znaleziono pliku: {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)
