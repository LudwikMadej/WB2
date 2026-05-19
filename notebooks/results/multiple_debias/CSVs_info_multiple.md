# multiple_debias/CONCEPT/recovery/ — recovery_summary_layer{L}_iter{N}.csv

Co mierzą: skuteczność iteracyjnego debiasingiu i stopień odbudowania konceptu przez późniejsze warstwy CLIP.

| Kolumna                | Znaczenie                                                              |
|------------------------|------------------------------------------------------------------------|
| `concept`              | nazwa debiasowanego konceptu                                           |
| `method`               | metoda CAV: `diff_means`, `lr`, `pclarc`                               |
| `debias_layer`         | warstwa, w której zastosowano debiasing                                |
| `n_iter`               | liczba iteracji debiasingiu                                            |
| `baseline_test`        | dokładność sondy przed debiasingiem (test set)                         |
| `after_{N}iter_test`   | dokładność sondy po N iteracjach (test set)                            |
| `max_recovery_test`    | maksymalna dokładność osiągnięta w kolejnych warstwach po debiasingiu  |
| `max_recovery_layer`   | warstwa, w której nastąpiło maksymalne odbudowanie konceptu            |

Interpretacja: `max_recovery_test` bliskie 1.0 przy `max_recovery_layer` > `debias_layer` oznacza, że późniejsze warstwy odbudowały usunięty koncept — debiasing w jednym miejscu nie jest trwały.
