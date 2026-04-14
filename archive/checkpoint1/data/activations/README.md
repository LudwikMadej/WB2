# Activation Vectors

This directory contains activations extracted from the CLIP model across different layers.

## Subdirectories

- **dm/**: Contains the activation vectors captured after the **Difference of Means (DM)** debiasing intervention.
- **lr/**: Contains the activation vectors captured after the **Logistic Regression (LR)** debiasing intervention.

Each CSV file in these folders (e.g., `layer_15.csv`) contains the 1024-dimensional [CLS] activation for every image in the test set and label column.
