# Fairness-diagnostics-utkface
Computational artefacts supporting a socio-technical analysis of algorithmic bias in facial classification. The repository contains diagnostic scripts demonstrating how aggregate performance metrics can obscure subgroup-level disparities under demographic imbalance.


## Script B — Gender Classification Pipeline  
**File:** `scripts/Gender_classification_pipeline.py`

### Purpose

This script implements a controlled gender classification pipeline on the UTKFace dataset in order to establish **aggregate-level performance metrics** prior to any subgroup or fairness-specific analysis. Its purpose is diagnostic rather than evaluative: it demonstrates how a facial analysis system can appear technically competent when assessed solely using overall accuracy and ROC/AUC.

---

### Dataset Conditions

- Dataset: UTKFace (public academic dataset)
- Task: Binary gender classification (as annotated in dataset filenames)
- Input resolution: 128 × 128 RGB images
- Train–validation split: 80% training / 20% validation
- Demographic attributes (race, age) are parsed but **not used for stratification**
- No data augmentation or bias mitigation techniques are applied

Demographic labels are treated strictly as dataset annotations rather than as socially or biologically definitive categories.

---

### What the Script Does

- Parses demographic metadata from UTKFace filenames
- Constructs a TensorFlow data pipeline with deterministic preprocessing
- Trains a lightweight convolutional neural network
- Evaluates model performance using:
  - Aggregate accuracy
  - Receiver Operating Characteristic (ROC) curve
  - Area Under the Curve (AUC)
- Produces a single ROC curve for the validation subset

---

### What the Script Does *Not* Claim

- It does **not** assess algorithmic fairness
- It does **not** measure subgroup-specific performance
- It does **not** claim demographic parity or bias mitigation
- It does **not** represent an operational or deployable facial recognition system
- It does **not** imply ethical acceptability or real-world suitability

High aggregate performance in this script should **not** be interpreted as evidence of fairness.

---

### How This Script Supports the Report

This script supports **Methodology Section 3.3.3 (Stage 1: Aggregate Evaluation)** by:

- Establishing a technical baseline against which subgroup disparities are later contrasted
- Demonstrating the masking effect of aggregate metrics on demographic risk
- Providing empirical grounding for the report’s critique of accuracy-centric evaluation practices in high-risk biometric systems

The results produced here are intentionally incomplete and are used only as a reference point for subsequent fairness diagnostics.

---

### Methodological Positioning

This script is **diagnostic rather than evaluative**.  
It is designed to illustrate **evaluation blind spots**, not to optimise performance or validate system effectiveness.

