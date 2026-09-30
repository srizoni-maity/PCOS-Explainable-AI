# Explainable Machine Learning for Survey-Based PCOS Risk Classification

<div align="center">

**Explainable Machine Learning for Survey-Based PCOS Risk Stratification Using Dietary, Lifestyle, and Trans-Fat-Related Information Among University-Age Women.**

<br>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0-189A3B?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00?style=for-the-badge)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=for-the-badge)](https://shap.readthedocs.io/)

</div>

---

## About the Project

Polycystic ovary syndrome (PCOS) is a complex condition influenced by multiple interacting factors, making risk assessment from isolated variables difficult. This project investigates whether **questionnaire-based dietary, lifestyle, symptom, awareness, and medical-history information can be used to identify patterns associated with PCOS risk** using explainable machine learning.
Rather than treating the questionnaire as a simple collection of independent variables, the framework explores different representations of the same information and examines how feature construction, feature selection, model choice, and explainability affect predictive performance.
Three feature representations are investigated:
* **RAW** — original questionnaire variables
* **COMPOSITE** — selected higher-level constructed features
* **HYBRID** — original and constructed information combined

The resulting pipeline combines **consensus-based feature ranking, feature-subset analysis, nested cross-validation, hyperparameter optimization, independent test evaluation, model explainability, calibration, bootstrap uncertainty analysis, and decision-curve analysis**.
The repository contains the computational implementation and supporting non-participant-level outputs associated with the study.
> **Research focus:** building a reproducible and interpretable ML pipeline for questionnaire-based PCOS risk classification — not a clinical diagnostic system.

---

## Framework

```text
                    Questionnaire Data
                           │
                           ▼
                 Data Quality Control
                           │
                           ▼
             ┌─────────────┼─────────────┐
             │             │             │
            RAW        COMPOSITE       HYBRID
             │             │             │
             └─────────────┼─────────────┘
                           ▼
               Consensus Feature Ranking
                           │
                           ▼
                Feature Subset Evaluation
                           │
                           ▼
                Nested Cross-Validation
                           │
                           ▼
                 Model Optimization
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
            LR             RF        XGBoost / CatBoost
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                Independent Test Set
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
          SHAP       Permutation /      Ablation
                       LOFO
            │
            └──────────────┬──────────────┘
                           ▼
              Calibration / Bootstrap
                    / Decision Curve
```

---

## What We Used

| Component                     | Methods / Tools                                          |
| ----------------------------- | -------------------------------------------------------- |
| **Data processing**           | Python, NumPy, pandas, SciPy                             |
| **Feature engineering**       | RAW / COMPOSITE / HYBRID representations                 |
| **Feature ranking**           | Mutual Information, ANOVA, Random Forest, LASSO, XGBoost |
| **Feature aggregation**       | Borda-count consensus ranking                            |
| **Models**                    | Logistic Regression, Random Forest, XGBoost, CatBoost    |
| **Validation**                | Nested Stratified Cross-Validation                       |
| **Optimization**              | Optuna                                                   |
| **Explainability**            | SHAP, Permutation Importance, LOFO, Ablation             |
| **Reliability analysis**      | Calibration, Bootstrap Uncertainty                       |
| **Clinical utility analysis** | Decision-Curve Analysis                                  |


---

## Reproducibility

The experiments use a fixed random seed and a structured training/evaluation pipeline separating model development from independent test evaluation.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

The repository provides the code and non-participant-level outputs needed to inspect the computational workflow.

---

## Data Availability

The participant-level survey dataset is **not publicly available** because of privacy and data-sharing considerations.
The dataset used in the study is **available from the corresponding author upon reasonable request**, subject to applicable privacy and data-sharing requirements.

**Corresponding author:**
Dr. Masihuddin
`masih@iisertvm.ac.in`

---

## Citation

If you use this repository or build upon this work, please cite the associated paper:
> **Explainable Machine Learning for Survey-Based PCOS Risk Classification Using Dietary, Lifestyle, and Trans-Fat-Related Information Among University-Age Women**

---

## Disclaimer
This repository is intended for **research and methodological purposes**. The proposed framework is a risk-classification approach and **is not a clinical diagnostic system or a substitute for professional medical assessment**.

---

<div align="center">

**Research code • Reproducible workflow • Explainable machine learning**

</div>
