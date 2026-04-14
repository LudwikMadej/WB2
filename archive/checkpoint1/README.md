# Checkpoint 1: Concept Debiasing Pipeline

This directory contains a modular pipeline for investigating and debiasing the "bald" concept in MetaCLIP (`clip-vit-large-patch14`).

## Pipeline Structure

1. **01_data_acquisition.ipynb**: Downloads CelebA images, filters for the target concept, resizes them (224x224), and saves them locally.
2. **02_activation_extraction.ipynb**: Extracts baseline activations for the training set at the intervention layer (Layer 15).
3. **03_cav_generation.ipynb**: Trains Concept Activation Vectors (CAVs) using Difference of Means and Logistic Regression.
4. **04_debiasing_inference.ipynb**: Performs the debiasing intervention (orthogonalization on all patches) and captures activations for subsequent layers.
5. **05_recovery_analysis.ipynb**: Evaluates the re-emergence of the concept in deeper layers using fresh classifiers.

## Data Organization

- `/data`: Contains images, metadata, and saved activation vectors.
- `/models`: Contains saved CAVs and the original probe model.
