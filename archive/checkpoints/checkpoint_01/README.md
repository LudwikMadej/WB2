# Checkpoint 01 - concept debiasing w CLIP ViT-L/14 na CelebA

Cel: wytrenowanie klasyfikatora binarnego wykrywającego dany koncept w aktywacjach CLIP, a następnie usunięcie go metodami CAV (ortogonalna projekcja).

## Struktura

- `bald/` - pierwsza iteracja: koncept łysiny, ekstrakcja aktywacji z warstwy 15, CAV LR i DM
- `eyeglasses/` - główna linia: koncept okularów, wszystkie 24 warstwy, klasyfikatory LR / XGBoost / TorchLR, metody debiasingu LR / DM / P-ClArC
