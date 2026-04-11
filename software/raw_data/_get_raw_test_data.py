import os
import glob
import pandas as pd

def get_raw_test_data(layer_index: int, data_folder_path='../data/') -> pd.DataFrame:
    tmp_path = os.path.join(data_folder_path, f'activations/test/layer_{layer_index:02d}.csv')

    return pd.read_csv(tmp_path)