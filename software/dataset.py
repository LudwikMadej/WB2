from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
_DEFAULT_METADATA = _ROOT / 'data' / 'metadata.csv'

DS_SIZE = 256


def get_concept_split(
    concept: str,
    split: str,
    *,
    ds_size: int = DS_SIZE,
    seed: int = 42,
    metadata_path=None,
) -> pd.DataFrame:
    """Return a balanced, reproducible subset of metadata rows for `concept` and `split`.

    Train: 7 * ds_size rows — (7 * ds_size // 2) positive and negative each.
    Test:      ds_size rows — (ds_size // 2) positive and negative each.

    The subset is deterministic: same (concept, split, ds_size, seed) always returns
    the same filenames. All metadata columns are returned (filename, split, all attributes).
    """
    if metadata_path is None:
        metadata_path = _DEFAULT_METADATA
    df = pd.read_csv(metadata_path)
    df = df[df['split'] == split].copy()
    if concept not in df.columns:
        raise ValueError(
            f'Concept "{concept}" not in metadata. '
            f'Available columns: {[c for c in df.columns if c not in ("filename", "split")]}'
        )

    n_per_class = (7 * ds_size // 2) if split == 'train' else (ds_size // 2)

    pos = df[df[concept] == 1]
    neg = df[df[concept] == 0]

    n_pos = min(n_per_class, len(pos))
    n_neg = min(n_per_class, len(neg))

    if n_pos < n_per_class:
        print(f'[dataset] Warning: only {len(pos)} positives for {concept}/{split} '
              f'(requested {n_per_class})')
    if n_neg < n_per_class:
        print(f'[dataset] Warning: only {len(neg)} negatives for {concept}/{split} '
              f'(requested {n_per_class})')

    pos_sample = pos.sample(n=n_pos, random_state=seed)
    neg_sample = neg.sample(n=n_neg, random_state=seed + 1)

    return pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
