# WB2 — CLIP concept debiasing

Debiasing konceptu okularów w CLIP ViT-L/14 na CelebA.

## Struktura

```
Notebooks/   # 01 loading → 02 activations → 03 detection → 04 debiasing
               # warianty *_torch.ipynb używają TorchLR (GPU) zamiast sklearn LR
software/    # helpery: get_raw_{train,test}_data(layer), torch_lr.py
scripts/     # jednorazowe narzędzia (migrate_csv_to_parquet.py)
data/        # obrazy CelebA, metadata, aktywacje (parquet, float16 + zstd)
plots/       # wykresy concept detection (sklearn i torch)
```

## Zmiany względem main

1. **Aktywacje: CSV → Parquet (float16 + zstd)** — `scripts/migrate_csv_to_parquet.py` skonwertował aktywacje do parquet ze stratną kompresją. Rozmiar danych ~10× mniejszy. Notatnik `02` zapisuje bezpośrednio w parquet.
2. **TorchLR** — `software/torch_lr.py`: binarna regresja logistyczna na GPU (LBFGS), drop-in zamiennik sklearn `LogisticRegression`. Identyczna funkcja celu (L2 + BCE).
3. **Notatniki torch** — `03_concepts_detection_torch.ipynb` i `04_concepts_detection_after_debiasing_torch.ipynb` używają `TorchLR` zamiast sklearn LR. XGBoost bez zmian. Wyniki w osobnych katalogach (`data/torch_lr_*`), nie nadpisują baseline'u sklearn.
4. **P-ClArC** — notatnik `04_*_torch.ipynb` dodaje trzecią metodę debiasingu (Projective Class Artifact Compensation) obok istniejących LR i DM.

## Instalacja

Projekt używa nowego standardu: **`pyproject.toml` (PEP 621) + `uv`**. `pyproject.toml` + `uv.lock` to źródło prawdy. `requirements.txt` jest dogenerowywany dla środowisk bez uv (Colab).

**Zalecane — `uv` (RunPod, lokalnie):**

```bash
pip install uv
uv sync --extra jupyter
.venv/bin/python -m ipykernel install --user --name wb2 --display-name "WB2"
```

Potem w notatniku: **Kernel → Change Kernel → WB2** → Restart.

Trzecia komenda rejestruje `.venv/` jako kernel Jupytera — bez niej RunPod odpali notatnik na systemowym Pythonie i dostaniesz `ModuleNotFoundError: No module named 'pandas'`.

**Alternatywnie — Colab (bez uv):**

```bash
pip install -r requirements.txt
```

`requirements.txt` ma na pierwszej linii `--extra-index-url` do `download.pytorch.org/whl/cu126`, więc pip znajdzie CUDA wheel torcha.

## Regeneracja `requirements.txt`

Po zmianie `pyproject.toml`:

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project \
  | (echo '--extra-index-url https://download.pytorch.org/whl/cu126'; cat) \
  > requirements.txt
```

Nie edytuj `requirements.txt` ręcznie.

## Kolejność notatników

`01_loading_data` → `02_getting_activations` → `03_concepts_detection` → `04_concepts_detection_after_debiasing`.

Warianty `*_torch.ipynb` dla 03 i 04 można uruchomić niezależnie od wersji sklearn — potrzebują tych samych danych z 01/02.

`01` wymaga `.env` z tokenem HuggingFace. `02` potrzebuje GPU.
