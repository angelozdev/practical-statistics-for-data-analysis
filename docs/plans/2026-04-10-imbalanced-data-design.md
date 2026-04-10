# Design: Strategies for Imbalanced Data (Notebook 22)

**Date:** 2026-04-10
**File:** `notebooks/22-imbalanced-data.ipynb`

## Goal

Create a practical notebook that teaches strategies for imbalanced data from zero to professional level, using industry-standard tools (sklearn + imbalanced-learn).

## Dataset

Synthetic fraud detection dataset generated with `sklearn.datasets.make_classification`:
- 5,000 transactions, ~3% fraud (~150 positive cases)
- Features: transaction amount, hour of day, distance to usual merchant, customer age
- Intuitive domain: everyone understands fraud is rare

## Tools

- `sklearn`: LogisticRegression, StandardScaler, classification_report, GridSearchCV, PCA
- `imbalanced-learn`: RandomUnderSampler, RandomOverSampler, SMOTE, Pipeline

## Structure

### Section 1 — La trampa del accuracy
- Train baseline `LogisticRegression` with no resampling strategy
- Show ~97% accuracy but near-zero recall for fraud class
- Use `classification_report` and confusion matrix to expose the problem
- Key insight: high accuracy ≠ useful model with imbalanced data

### Section 2 — Undersampling (Downsampling)
- Concept: reduce majority class to match minority class size
- Implementation: `RandomUnderSampler` from imblearn
- Trade-off: improved recall but lose information from majority class
- Show before/after class distribution

### Section 3 — Oversampling (Upsampling)
- Concept: bootstrap minority class to match majority class size
- Implementation: `RandomOverSampler` from imblearn
- Trade-off: risk of overfitting (exact duplicate points)
- Show before/after class distribution

### Section 4 — Class Weights (Up-weight/Down-weight)
- Concept: penalize errors on minority class more heavily, no data modification
- Implementation: `class_weight='balanced'` in LogisticRegression
- Simplest to deploy in production
- Trade-off: less control than resampling

### Section 5 — SMOTE (Data Generation)
- First explain z-score / standardization (required because SMOTE uses KNN distances)
- Concept: generate synthetic minority examples by interpolating between K nearest neighbors
- Implementation: `SMOTE` from imblearn with `StandardScaler` preprocessing
- Visualization: PCA 2D plot comparing original vs SMOTE-generated points
- Key terms: z-score, K neighbors
- Trade-off: can introduce noise if class clusters overlap

### Section 6 — Comparative Table
- DataFrame with all strategies vs metrics: Precision, Recall, F1-score, ROC-AUC
- Guided conclusion: when to use each strategy

### Section 7 — Professional Pipeline
- Use `imblearn.pipeline.Pipeline` (not sklearn's — supports resamplers mid-pipeline)
- Full pipeline: `StandardScaler → SMOTE → LogisticRegression`
- `GridSearchCV` tuning `SMOTE__k_neighbors` and `LogisticRegression__C`
- Production-ready pattern

## Success Criteria

- Reader understands why accuracy is a misleading metric for imbalanced data
- Reader can apply each of the 4 strategies independently
- Reader understands the intuition behind SMOTE (z-score + KNN interpolation)
- Reader has a production-ready pipeline template
