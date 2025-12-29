# Baseline CNN Fairness Diagnostics (UTKFace)

## Purpose
This script implements a capacity-limited baseline convolutional neural network (CNN) to illustrate how algorithmic bias can emerge under demographic imbalance and weak inductive assumptions.

The script is **diagnostic rather than evaluative**. It does not seek to optimise performance or propose a deployable facial recognition system.

## Dataset Conditions
The script operates on the UTKFace dataset using filename-encoded demographic metadata (age, gender, race).  
Only the `UTKFace/` directory is used. Images are not redistributed or uploaded to this repository.

Demographic labels are treated as dataset annotations rather than ontological truths.

## What This Script Does
- Trains a simple CNN for binary gender classification
- Computes aggregate accuracy and ROC/AUC
- Evaluates subgroup performance disaggregated by:
  - gender
  - race
  - age group
- Visualises demographic distributions within the validation subset

## What This Script Does NOT Claim
- It does not claim real-world policing validity
- It does not benchmark state-of-the-art performance
- It does not infer causal sources of bias
- It does not evaluate individual identity recognition

## How This Script Supports the Report
This script supports the methodological claim that **aggregate performance metrics can conceal subgroup-specific error disparities**, particularly under demographically imbalanced conditions.

The results are used to:
- establish a weak baseline
- motivate architectural comparison
- demonstrate evaluation blind spots discussed in the fairness literature

The script corresponds directly to:
> Methodology Section 3.3.3 — Model Training and Bias Illustration  
> Figures 3.4, 3.6, 3.7, and 3.8

## Reproducibility Notes
Results may vary slightly due to random initialisation and dataset shuffling. All preprocessing, splits, and evaluation metrics are explicitly defined in the script.
