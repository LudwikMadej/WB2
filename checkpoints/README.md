# Checkpoints

## checkpoint_01 - concept debiasing w CLIP ViT-L/14 na CelebA

Dwie ścieżki eksperymentów:

- `bald/` - prototyp na koncept łysiny, warstwa 15
- `eyeglasses/` - główna linia: koncept okularów, wszystkie 24 warstwy, debiasing

## checkpoint_02 - sequential debiasing & concept recovery

- `01_cav_debiasing.ipynb` - sekwencyjny debiasing 24 warstw, 3 metody CAV (diff_means, lr, pclarc)
- `02_concept_recovery.ipynb` - multi-koncept × multi-metoda; bada powrót konceptu po jednorazowym debiasingu jednej warstwy
- `00_sequential_pclark_debiasing.ipynb` - wcześniejszy prototyp P-ClArC (Colab-style), zastąpiony przez `01_*`
