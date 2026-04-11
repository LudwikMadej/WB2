import os
import glob
import pandas as pd

def get_raw_train_data(layer_index: int, data_folder_path='../data/') -> pd.DataFrame:
    tmp_path = os.path.join(data_folder_path, 'activations/train/raw_batched/')
    
    all_files = os.listdir(tmp_path)
    if not all_files:
        print(f"Folder {tmp_path} jest pusty.")
        return pd.DataFrame()

    layer_ids = sorted(list(set([f.split('_')[0] for f in all_files])))

    # Sprawdzenie, czy indeks mieści się w zakresie
    if layer_index >= 24:
        print(f"Błąd: layer_index {layer_index} jest poza zakresem (dostępnych warstw: {len(layer_ids)}).")
        return pd.DataFrame()

    # Wybieramy konkretne ID na podstawie indeksu
    target_layer_id = layer_ids[layer_index]
    
    # Budujemy wzorzec wyszukiwania plików dla tej konkretnej warstwy
    search_pattern = os.path.join(tmp_path, f"{target_layer_id}_*.csv")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"Nie znaleziono plików dla warstwy o ID: {target_layer_id}")
        return pd.DataFrame()

    # Łączenie plików w jedną ramkę
    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    return combined