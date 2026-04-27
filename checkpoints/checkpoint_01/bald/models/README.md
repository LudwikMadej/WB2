# Model Artifacts

This directory contains Concept Activation Vectors (CAVs) and saved model probes.

## Subdirectories

- **cavs/**: Contains the serialized CAV vectors and the trained Logistic Regression probe.

## Key Files

- `cav_dm.npy`: The Concept Activation Vector calculated via Difference of Means.
- `cav_lr.npy`: The Concept Activation Vector calculated via Logistic Regression coefficients.
- `lr_model.pkl`: The serialized scikit-learn Logistic Regression model used as the "original probe".
