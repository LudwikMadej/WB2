# Checkpoint 02 - sequential debiasing & concept recovery

CLIP ViT-L/14, koncepty z CelebA. Eksperymenty z usuwaniem konceptu warstwa-po-warstwie i z badaniem, czy koncept "wraca" w kolejnych warstwach po jednorazowym debiasingu w środku sieci.

## Notatniki

### `01_cav_debiasing.ipynb` - sekwencyjny debiasing wszystkich warstw

Dla pojedynczego konceptu (`CONCEPT`) i wszystkich 24 warstw enkodera vision:

1. Wyciąga aktywacje CLS warstwa po warstwie.
2. Na każdej warstwie wyznacza CAV trzema metodami: `diff_means`, `lr` (LogisticRegressionCV po C), `pclarc`.
3. Mierzy accuracy CAV (próg/sklasyfikator nauczone na trainie, oceniane na trainie i teście).
4. Każda kolejna warstwa otrzymuje aktywacje już zdebiesowane na wszystkich poprzednich (forward hooki na CLS).
5. Zapisuje aktywacje pre/post (`layer_XX.parquet`) i CAV-y + metryki (`cavs.csv`) - osobno per metoda.

Output: `data/checkpoint2_data/{CONCEPT}/{method}/{cavs.csv, activations/layer_XX.parquet}`.

### `02_concept_recovery.ipynb` - odzyskiwanie konceptu po debiasingu jednej warstwy

Multi-koncept × multi-metoda, **jedno uruchomienie** dla wszystkich par `(concept, method)`. Konfiguracja: listy `CONCEPTS`, `METHODS`, parametry `DEBIAS_LAYER`, `N_ITER`.

Dla każdej pary:

1. Ekstrakcja CLS z `DEBIAS_LAYER` (raz na koncept, dzielona przez 3 metody).
2. Iteracyjny debiasing (`N_ITER` razy) - każda iteracja: rzut z poprzedniego CAV, nowy CAV, pomiar. Konwencja: `iter k = stan po k zastosowanych rzutach`.
3. Wstrzyknięcie zdebiesowanego CLS z powrotem do CLIP w `DEBIAS_LAYER`; reszta sieci działa bez debiasingu.
4. Zbieranie CLS z warstw `DEBIAS_LAYER+1 .. NUM_LAYERS-1` i trening LR (CV po C) na każdej z nich → krzywa "powrotu konceptu".

Output:

- `data/checkpoint2_data/{concept}/recovery/layer{L}_{method}_iter{N}/{iter_debiasing_accuracy.csv, layer_recovery_accuracy.csv}`
- per koncept: `data/checkpoint2_data/{concept}/recovery/comparison_layer{L}_iter{N}.png` (3 metody na jednym wykresie)
- zbiorczy: `data/checkpoint2_data/recovery_summary_layer{L}_iter{N}.csv`

### `00_sequential_pclark_debiasing.ipynb` - wcześniejszy prototyp

Pierwsze podejście do sekwencyjnego P-ClArC debiasingu (Colab-style, wymaga ręcznego sklonowania repo). Funkcjonalnie zastąpione przez `01_cav_debiasing.ipynb`. Wyniki w `sequential_debiasing_results/experiments_results/`.

## Parametry eksperymentu

- `MODEL_ID = openai/clip-vit-large-patch14` (24 warstwy enkodera vision, D=1024, bf16 na GPU)
- `LR_CS = [0.01, 0.1]`, `LR_CV_FOLDS = 5` - dobór L2 dla LR przez cross-validation
- Domyślnie 1700 train / 300 test próbek na koncept (zbalansowane na cechę przez `scripts/download_data.ipynb`)

## Założenia metodologiczne

- **CAV diff_means** kieruje neg→pos z konstrukcji; znak progu deterministyczny.
- **CAV lr** wyznaczany w przestrzeni standaryzowanej, przeskalowany z powrotem przez `coef_/scale_`. Hook ortogonalny w przestrzeni oryginalnej w pełni zeruje logit LR (dowód: rzut na gradient w przestrzeni oryginalnej daje `(w/std)·x_new = 0`).
- **PCLARC** target = średnia projekcji klasy negatywnej na trainie. Zmienia rozkład projekcji jednostronnie. Po jednej iteracji średnie klas są niemal identyczne - kolejne iteracje są degenerujące, więc dla `pclarc` praktycznie używaj `N_ITER=1`.
- **Sequential debiasing nie jest gwarantem braku konceptu w kolejnych warstwach**: nieliniowości attention + MLP + LayerNorm mogą rekonstruować kierunek. To przedmiot eksperymentu w `02_*`, nie bug.
- Ocena testowa używa progów/sklasyfikatorów wyznaczonych wyłącznie na trainie (brak data leakage).
