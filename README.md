# Replication Package for Method-Level Energy Prediction in Java Using Code Features and Execution Time

This repository contains the replication package for our study on estimating power consumption of Java methods using static code features. It includes all necessary scripts and setup required to replicate our experiments.

We investigate whether lightweight static and dynamic code metrics can reliably predict method-level energy usage. The package includes the machine learning pipeline, dataset, and supporting scripts to reproduce our evaluation.

---

## 1. Requirements

- **Java 17**  
- **Python 3.10.5**  
- JVM energy profiler: [JoularJX](https://www.noureddine.org/research/joular/joularjx)  
- Java performance profiler: [async-profiler](https://github.com/async-profiler/async-profiler)
- Python dependencies listed in `requirements.txt`

### Setup

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Directory Structure and Description

This package is structured as follows:

```plaintext
replication-package_method_energy_java/
│── data/                    # Contains raw and processed datasets and subject Java soruce code
│── scripts/                 # Python scripts for subject files extraction, feature extraction, pre and post processing
│── ML_prediction/           # ML pipeline for energy prediction
├── static_analysis/         # Static analyzers and AST-based metric extractors
│── outputs/                 # Contains energy and performance profiling reports and 
│── requirements.txt         # Python dependencies
│── README.md                # This file
```

## Folder Descriptions
- data/: Includes the CSV files used in the ML pipeline containing static features, dynamic energy values, and additional method metadata. Java subjects are also provided for reference.

- scripts/: Contains helper scripts for subject-level preprocessing, file parsing, filtering, and code property extraction.

- static_analysis/: Implements the static metric collection logic (e.g., AST-based depth, cyclomatic complexity, loop counts). Used to generate features in earlier stages of the study.

- ML_prediction/: Reproducible end-to-end pipeline to train, evaluate, and visualize machine learning models for predicting energy consumption. Includes baseline models, feature selection, hyperparameter tuning, and cross-validation routines.

### 3. Replication Scope
This package enables replication of the machine learning phase of our study.
The dynamic energy measurement phase was performed using JoularJX and Async-Profiler, and while not included here, the resulting dataset is provided ```(data/method_dataset_*.csv)```.

The following components are fully reproducible:
- Feature selection pipelines
- Training and evaluation of ML regressors
- SHAP importance analysis
- Aggregated results and visualizations
- Top configuration selection and performance trade-off plots

For experiment steps and methodology, please refer to our paper.


