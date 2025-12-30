## Script D: Dataset Demographic Diagnostics (UTKFace)
**File:** scripts/Subgroup_Aggregate_Metric_Age_Race_Gender.py
**Branch:** dataset-demographic-diagnostic

## Purpose
This script performs a dataset-level demographic diagnostic to visualise representational imbalance across race, gender, and age in the UTKFace dataset prior to any model training or evaluation.

The script is descriptive rather than predictive. It establishes structural conditions under which algorithmic bias may emerge, rather than measuring algorithmic performance.

## Dataset Conditions
Dataset: UTKFace (public academic dataset)
Scope: Full dataset (23,705 images)

Demographic attributes parsed from filename metadata:
- age
- binary gender
- race (five-category label)
- No train–validation split is applied
- No modelling, optimisation, or prediction occurs

Demographic labels are treated strictly as dataset annotations, not as socially or biologically definitive categories.

## What This Script Does
- Parses demographic metadata from UTKFace filenames
- Quantifies and visualises:
 - Race distribution (bar chart)
 - Gender distribution (bar chart)
 - Age distribution (histogram)
- Produces descriptive figures illustrating dataset skew

## What This Script Does NOT Claim
This script does not:
- Train or evaluate any machine learning model
- Measure algorithmic bias or fairness outcomes
- Attribute causality to observed demographic imbalance
- Represent real-world population distributions
- Assess ethical acceptability or deployment suitability

## How This Script Supports the Report
This script supports 
> Methodology Section 3.3.2 (Visualisation of Demographic Imbalance) by providing empirical grounding for subsequent bias diagnostics.

The figures produced here:
- Demonstrate representational skew as a precondition for bias
- Contextualise subgroup performance disparities observed in later scripts
- Anchor fairness analysis in dataset structure rather than model behaviour alone

The outputs correspond directly to:
> Figures 3.1–3.3 — Dataset demographic distributions

## Methodological Positioning
This script operationalises a socio-technical framing of bias, treating dataset composition as an upstream governance and design issue rather than a downstream modelling defect.


## Script A: Baseline CNN Fairness Diagnostics (UTKFace)
**File:** `scripts/Baseline_CNN_ROC_Fairness_metrics.py`
**Branch:** `baseline-cnn-fairness`

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


## Script B: Gender Classification Pipeline  
**File:** `scripts/Gender_classification_pipeline.py`
**Branch:** `gender-pipeline-diagnostics`

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



## Script C: MobileNetV2 Fairness Diagnostic (Architectural Comparison)

**File:** `scripts/MobileNetV2.py`  
**Branch:** `mobilenetv2-fairness-comparison`

### Purpose
This script implements a lightweight MobileNetV2-based gender classification pipeline to examine how architectural improvements affect aggregate performance and subgroup-specific error patterns under identical dataset conditions. The objective is diagnostic rather than optimising: the script is used to illustrate whether increased representational capacity and pretrained feature extraction reduce—but do not eliminate—demographic performance disparities observed in baseline models.

### Dataset Conditions
The script uses the UTKFace dataset under the same constraints applied throughout the project:
- A fixed 80/20 train–validation split
- Gender classification as a proxy task
- Demographic metadata (age, gender, race) parsed from filenames
- No identity recognition or individual-level inference
- No dataset augmentation or rebalancing

MobileNetV2 is initialised with ImageNet-pretrained weights, and all base layers are frozen. This ensures architectural comparison without introducing confounding optimisation effects.

### What This Script Does NOT Claim
This script does **not**:
- Claim suitability for operational policing or forensic deployment
- Demonstrate fairness, bias elimination, or regulatory compliance
- Establish causal fairness improvements beyond this dataset
- Benchmark or rank facial recognition systems
- Generalise findings beyond controlled experimental conditions

Observed performance changes are interpreted as illustrative mechanisms rather than normative guarantees.

### How This Script Supports the Report
Script C supports **Methodology Section 3.3.3 (Model Training and Bias Illustration)** by providing a contrastive architecture to the baseline CNN. It empirically demonstrates that:
- Aggregate accuracy and ROC/AUC may improve with architectural sophistication
- Subgroup disparities persist even when overall performance stabilises
- Technical correction can mitigate but not resolve structural bias

These findings are used to reinforce the report’s central claim that algorithmic bias is a socio-technical phenomenon shaped by dataset composition, evaluation practices, and institutional interpretation, rather than a defect solvable through architecture alone.


