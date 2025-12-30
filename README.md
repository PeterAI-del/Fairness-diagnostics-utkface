## Script C: MobileNetV2 Fairness Diagnostic (Architectural Comparison)

**File:** `scripts/MobileNetV2_fairness_comparison.py`  
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

