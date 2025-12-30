# Fairness-diagnostics-utkface
Computational artefacts supporting a socio-technical analysis of algorithmic bias in facial classification. The repository contains diagnostic scripts demonstrating how aggregate performance metrics can obscure subgroup-level disparities under demographic imbalance.
## Script D: Dataset-Level Demographic Imbalance Diagnostics

**File:** `scripts/Subgroup_Aggregate_Metric_Age_Race_Gender.py`  
**Branch:** `dataset-demographic-diagnostics`

### Purpose
This script provides descriptive, dataset-level diagnostics of demographic composition within the UTKFace dataset. It visualises race, gender, and age distributions across the full dataset to empirically demonstrate representational imbalance prior to any model training or evaluation.

The script establishes that observed subgroup disparities in later analyses are structurally prefigured by dataset composition rather than artefacts of model architecture or optimisation.

### Dataset Conditions
- Full UTKFace dataset (23,705 images)
- No train–validation split
- No model training or inference
- Demographic attributes parsed from filenames
- Race, gender, and age treated as dataset annotations rather than ontological categories

### What This Script Does NOT Claim
This script does **not**:
- Evaluate model performance or fairness
- Attribute bias to any specific algorithm
- Infer causal relationships between demographics and outcomes
- Assess deployment suitability or legal compliance

It is strictly descriptive and diagnostic.

### How This Script Supports the Report
Script D supports **Methodology Section 3.3.2 (Visualisation of Demographic Imbalance)** by empirically grounding claims of representational skew. It provides the evidential basis for interpreting subgroup error asymmetries observed in subsequent baseline CNN and MobileNetV2 analyses.

By isolating dataset composition from modelling choices, the script reinforces the report’s central claim that algorithmic bias emerges from socio-technical pipelines rather than isolated technical defects.
