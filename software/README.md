# software

Współdzielony kod pomocniczy używany przez notatniki.

## Moduły

- `raw_data/` - funkcje `get_raw_train_data(layer_index)` i `get_raw_test_data(layer_index)` do wczytywania aktywacji z parquetu; ścieżka do `data/` obliczana automatycznie względem tego pakietu
- `torch_lr.py` - `TorchLR`: binarna regresja logistyczna na GPU (LBFGS), drop-in zamiennik sklearn `LogisticRegression`
