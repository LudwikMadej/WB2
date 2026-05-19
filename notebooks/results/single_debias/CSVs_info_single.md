# single_debias/CONCEPT/concept_detection/ — pliki 00–23_results.csv

Co mierzą: jak dobrze prosta sonda (LogisticRegression lub XGBoost) wykrywa koncept w aktywacjach każdej z 24 warstw CLIP, przed debiasingiem.

Każdy plik = jedna warstwa. Każda warstwa testowana 10 razy (bootstrap).

| Kolumna          | Znaczenie                              |
|------------------|----------------------------------------|
| `layer_id`       | numer warstwy (0–23)                   |
| `model`          | `LogisticRegression` lub `XGBoost`     |
| `run_id`         | numer bootstrapu (0–9)                 |
| `method`         | patrz niżej                            |
| `train_*` / `test_*` | metryki na zbiorze train/test     |

Trzy wartości `method`:

| Wartość               | Znaczenie                                           |
|-----------------------|-----------------------------------------------------|
| `real`                | prawdziwe etykiety — właściwy sygnał                |
| `random_shuffle`      | przetasowane etykiety — kontrola (~losowe)          |
| `random_unit_vector`  | losowy kierunek w przestrzeni — kontrola (~losowe)  |

Interpretacja: jeśli `test_roc_auc` dla `real` wynosi np. 0.75, a dla `random_shuffle` ~0.5 — sonda rzeczywiście wykrywa koncept, nie przypadkowy wzorzec. Im wyższy AUC w głębszych warstwach, tym bardziej CLIP tam koduje ten koncept.

---

# single_debias/eyeglasses/debiased_detection/ — pliki 00–23_debiased_results.csv

Co mierzą: to samo co wyżej, ale po debiasingiu w danej warstwie.

Dodatkowe kolumny:

| Kolumna         | Znaczenie                                                      |
|-----------------|----------------------------------------------------------------|
| `debias_method` | `lr` (regresja logistyczna jako CAV) lub `dm` (diff_means)     |

Interpretacja: porównujesz AUC przed i po. Jeśli debiasing zadziałał, `test_roc_auc` dla `real` powinno spaść — sonda nie może już wykryć konceptu w danej warstwie.
