# zero_shot_eval/eyeglasses/results.csv

Co mierzą: jak CLIP klasyfikuje zero-shot 40 atrybutów CelebA (np. male, smiling, bald…), przed i po debiasingiu okularów. Kluczowa kolumna: `delta_ba = BA_po − BA_przed`.

| Kolumna            | Znaczenie                                                              |
|--------------------|------------------------------------------------------------------------|
| `target_attribute` | jeden z 40 atrybutów CelebA                                            |
| `bal_acc`          | balanced accuracy po debiasingiu                                       |
| `auc`              | AUC-ROC                                                                |
| `regime`           | `raw` / `single` / `sequential` / `iterative` / `fixed_multi`         |
| `debiased_concept` | `none` lub nazwa konceptu                                              |
| `method`           | `diff_means`, `lr`, `pclarc`                                           |
| `debias_layer`     | warstwa gdzie zastosowano debiasing                                    |
| `ba_raw`           | BA bez debiasingiu (punkt odniesienia)                                 |
| `delta_ba`         | zmiana BA — główny wynik                                               |
| `n_iterations`     | liczba iteracji (tylko dla `iterative`)                                |
| `n_layers`         | liczba warstw (tylko dla `fixed_multi`)                                |

Baseline `raw` (bez debiasingiu):

| Atrybut      | BA    | Uwaga                          |
|--------------|-------|--------------------------------|
| `male`       | 0.986 | najlepiej klasyfikowany        |
| `eyeglasses` | 0.893 | debiasowany koncept            |

Co sprawdzasz przez `delta_ba`:

- Ujemny `delta_ba` dla `eyeglasses` = debiasing zadziałał (CLIP przestał "widzieć" okulary)
- Ujemny `delta_ba` dla innych atrybutów = collateral damage — debiasing niecelowo zaszkodził innym cechom (np. klasyfikacji płci, bo okulary korelują z płcią w danych)

Wykresy w tym folderze (`delta_ba_debiased_concept.png`, `heatmap_delta_ba_single_*.png`, `delta_ba_all_attrs_sequential.png`) wizualizują właśnie te efekty.
