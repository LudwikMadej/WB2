import os
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent.parent / 'data'

def get_raw_test_data(layer_index: int, data_folder_path=None) -> pd.DataFrame:
    if data_folder_path is None:
        data_folder_path = _DATA_DIR
    path = os.path.join(data_folder_path, f'activations/raw/test/layer_{layer_index:02d}.parquet')
    return pd.read_parquet(path)
