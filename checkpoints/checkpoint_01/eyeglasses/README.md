# Checkpoint 01 — eyeglasses

Główna linia eksperymentów CP1. Koncept okularów, wszystkie 24 warstwy CLIP ViT-L/14.

## Notatniki

- `01_loading_data` — pobieranie CelebA z HuggingFace, filtrowanie po okularach, zapis obrazów i metadanych
- `02_getting_activations` — ekstrakcja aktywacji GPU (wszystkie 24 warstwy), zapis do parquet
- `03_concepts_detection` — trening klasyfikatorów (LR, XGBoost) per warstwa, bootstrap, wizualizacja
- `03_concepts_detection_torch` — wariant z TorchLR (GPU LBFGS) zamiast sklearn LR
- `04_concepts_detection_after_debiasing` — debiasing przez CAV LR i CAV DM, ocena po debiasingu
- `04_concepts_detection_after_debiasing_torch` — wariant torch, dodatkowo P-ClArC
- `Instrukcja_obsługi_modelu` — demo hookowania warstw CLIP i podmiany aktywacji

## Kolejność

`01` → `02` → `03` (lub `03_torch`) → `04` (lub `04_torch`)

Warianty `_torch` można uruchomić niezależnie od wersji sklearn — potrzebują tych samych danych z `01`/`02`.

## Uwagi

- `01` wymaga `.env` z tokenem HuggingFace
- `02` wymaga GPU
- wyniki zapisywane do `data/` w korzeniu repo
- wykresy zapisywane do `plots/` w tym folderze
