# Replication Package for Estimating Method-Level Energy Usage in Java Benchmarks from Static Code Features

This repository contains the replication package for our study on estimating power consumption of Java methods using static code features. It includes all necessary scripts and setup required to replicate our experiments.

---

## 1. Requirements

- **Java 17**  
- **Python 3.10.5**  
- JVM energy profiler: [JoularJX](https://www.noureddine.org/research/joular/joularjx)  
- Java perforamnce profiler: [async-profiler](https://github.com/async-profiler/async-profiler)
- Python dependencies listed in `requirements.txt`

### Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt


## 2. Directory Structure and Description

This package is structured as follows:

```plaintext
replication-package_method_energy_java/
│── data/                    # Contains raw and processed datasets and subject Java soruce code
│── scripts/                 # Python scripts for subject files extraction, feature extraction, pre and post processing
│── ML_prediction/           # ML pipeline for energy prediction
│── outputs/                 # Contains energy and performance profiling reports and 
│── requirements.txt         # Python dependencies
│── README.md                # This file
```
