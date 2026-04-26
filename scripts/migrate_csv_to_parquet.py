"""Migracja aktywacji CSV -> parquet (float16 + zstd).

Operacje:
  - data/activations/test/layer_XX.csv -> data/activations/test/layer_XX.parquet
  - data/activations/train/raw_batched/XX_YY.csv -> scalenie -> data/activations/train/layer_XX.parquet

Po sukcesie usuwa oryginalne CSV. Uruchamiać z roota repo.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd

COMPRESSION = "zstd"
DTYPE = np.float16
TEST_DIR = "data/activations/test"
TRAIN_BATCH_DIR = "data/activations/train/raw_batched"
TRAIN_OUT_DIR = "data/activations/train"
N_LAYERS = 24
LABEL_COL = "glasses"


def convert_df(df: pd.DataFrame) -> pd.DataFrame:
    label = df[LABEL_COL].astype(np.int8)
    acts = df.drop(columns=[LABEL_COL]).astype(DTYPE)
    acts.columns = acts.columns.astype(str)
    acts[LABEL_COL] = label
    return acts


def migrate_test():
    print("\n=== TEST ===")
    for layer in range(N_LAYERS):
        src = os.path.join(TEST_DIR, f"layer_{layer:02d}.csv")
        dst = os.path.join(TEST_DIR, f"layer_{layer:02d}.parquet")
        if not os.path.exists(src):
            print(f"  [skip] brak {src}")
            continue
        df = pd.read_csv(src)
        out = convert_df(df)
        out.to_parquet(dst, compression=COMPRESSION, index=False)
        src_mb = os.path.getsize(src) / 1024**2
        dst_mb = os.path.getsize(dst) / 1024**2
        print(f"  layer_{layer:02d}: {src_mb:.1f} MB -> {dst_mb:.2f} MB  ({src_mb/dst_mb:.1f}x)")
        os.remove(src)


def migrate_train():
    print("\n=== TRAIN (scalam batche -> jeden parquet per warstwa) ===")
    os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
    for layer in range(N_LAYERS):
        pattern = os.path.join(TRAIN_BATCH_DIR, f"{layer:02d}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [skip] brak batchy dla warstwy {layer}")
            continue
        src_mb = sum(os.path.getsize(f) for f in files) / 1024**2
        combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out = convert_df(combined)
        dst = os.path.join(TRAIN_OUT_DIR, f"layer_{layer:02d}.parquet")
        out.to_parquet(dst, compression=COMPRESSION, index=False)
        dst_mb = os.path.getsize(dst) / 1024**2
        print(f"  layer_{layer:02d}: {len(files)} batchy, {src_mb:.1f} MB -> {dst_mb:.2f} MB  ({src_mb/dst_mb:.1f}x) [{len(combined)} obs]")
        for f in files:
            os.remove(f)

    if os.path.isdir(TRAIN_BATCH_DIR) and not os.listdir(TRAIN_BATCH_DIR):
        os.rmdir(TRAIN_BATCH_DIR)
        print(f"  usunięto pusty folder {TRAIN_BATCH_DIR}")


if __name__ == "__main__":
    if not os.path.isdir("data/activations"):
        print("Uruchom z roota repo (nie widzę data/activations/).", file=sys.stderr)
        sys.exit(1)

    migrate_test()
    migrate_train()
    print("\nGotowe.")
