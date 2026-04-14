import os
import pandas as pd

def get_raw_test_data(layer_index: int, data_folder_path='../data/') -> pd.DataFrame:
    path = os.path.join(data_folder_path, f'activations/test/layer_{layer_index:02d}.parquet')
    return pd.read_parquet(path)
