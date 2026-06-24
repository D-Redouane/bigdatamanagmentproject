
# MIMIC-IV Spark ML Project - Complete Guide

A production-ready implementation demonstrating Big Data engineering and machine learning workflows on clinical dataset paths using Apache Spark (PySpark) and Streamlit. This repository contains the complete pipeline for processing relational health records, training distributed ML models, extracting target parameters, and running a lightweight, responsive predictive UI context.

## 📋 Table of Contents
1. [Project Architecture](#project-architecture)
2. [Prerequisites & Requirements](#prerequisites--requirements)
3. [Local Installation & Setup](#local-installation--setup)
4. [Windows Environment Setup (Critical Fix)](#windows-environment-setup-critical-fix)
5. [Model Pipeline Architecture](#model-pipeline-architecture)
6. [Running the Application](#running-the-application)
7. [The Accuracy Framework & Benchmarks](#the-accuracy-framework--benchmarks)
8. [Troubleshooting](#troubleshooting)



## Project Architecture

```text
├── saved_spark_models/
│   └── model_metadata.json   # Exported mathematical weights & layout matrices
├── mimic-iv-clinical-database-demo-2.2/ # Local MIMIC source directory
├── app.py                     # 100% Native parameters Streamlit UI Dashboard
├── MIMIC_IV.ipynb             # Engineering, EDA, and Model Pipeline Notebook
└── README.md                  # Comprehensive System Guide

```

---

## Prerequisites & Requirements

Ensure you have the following environments pre-installed on your system machine:

* **Python**: `3.9` up to `3.11` (Note: `pyspark` compatibility anomalies can occur on Python `3.12+` depending on your setup).
* **Java Development Kit (JDK)**: Version `8` or `11` (Required by Apache Spark's Java Virtual Machine context).

---

## Local Installation & Setup

Follow these commands sequentially to initialize your working project directory, construct isolated virtual environments, and install necessary computational wheels.

### 1. Initialize Project & Environments

```bash
# Clone the repository
git clone [https://github.com/D-Redouane/bigdatamanagmentproject.git](https://github.com/D-Redouane/bigdatamanagmentproject.git)
cd bigdatamanagmentproject

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 2. Install Engine Dependencies

```bash
# Upgrade package installers
pip install --upgrade pip

# Install required numerical libraries, PySpark, and UI frameworks
pip install pyspark pandas numpy streamlit matplotlib seaborn

```

### 3. Verify Java Setup Context

Verify that your system path references a valid Java Runtime version:

```bash
java -version

```

---

## Windows Environment Setup (Critical Fix)

When running Apache Spark locally on a Windows platform, saving operational distributed model objects using native `.save()` interfaces natively forces a call out to Hadoop filesystem abstractions, throwing a fatal `Py4JJavaError` caused by missing `winutils.exe` binaries.

### The Pure Code Solution

To bypass this operating system constraint completely while removing all Scikit-Learn requirements, this codebase **does not serialize heavy file pointers via Spark**. Instead, the notebook pipeline directly extracts the matrix arrays and parameters straight out of the active JVM memory context and writes them directly into an optimized `model_metadata.json` configuration file.

This means you do **not** need to install complex system variables or global dependencies to run either the notebook calculations or the UI application layout on your local machine.

---

## Model Pipeline Architecture

The machine learning system trains three distinct estimators across partitioned training rows (`70% Train / 30% Test`):

1. **Multi-Class Logistic Regression**: Extracts raw structural coefficient intercepts and multi-tier weights matrices.
2. **Decision Tree Classifier**: Maps relational clinical branch splits down to descriptive logical configurations.
3. **Random Forest Classifier**: Processes structured ensembles to establish clear overall feature weight distributions.

### Feature Weights Extracted Natively

* `age_at_admission`: **28.7%**
* `num_diagnoses`: **26.4%**
* `length_of_stay`: **21.7%**
* `anchor_age`: **18.9%**
* `gender_encoded`: **4.3%**

---

## Running the Application

### 1. Process Data & Generate Weights

Open your Jupyter instance to run the model pipelines:

```bash
jupyter notebook MIMIC_IV.ipynb

```

Execute the cells from top to bottom. The notebook automatically imports clinical structures, manages target data vector transformation pipelines, fits the model context objects, and outputs the mathematical models to `saved_spark_models/model_metadata.json`.

### 2. Launch the Streamlit Analytics Interface

Once your `model_metadata.json` is generated, run the responsive web layout via standard execution shells:

```bash
streamlit run app.py

```

This serves up an isolated web container locally at `http://localhost:8501`.

The user interface operates with **zero Scikit-Learn overhead** and avoids initializing redundant local single-node master clusters during active user updates by compiling math transformations via direct NumPy vector dot-products and internal tree-string regex parsers.

---

## The Accuracy Framework & Benchmarks

The baseline classification operates over an intricate 25-way categorical prediction model with severe target imbalances (up to 37:1 ratios), turning a standard model run into an advanced machine learning challenge.

| Engine Platform | Observed Accuracy | vs. Random Baseline (4%) |
| --- | --- | --- |
| Random Guessing Baseline | 4.00% | Baseline |
| Majority Class Strategy | 13.40% | Baseline |
| **Decision Tree Model** | 9.86% | +147% |
| **Random Forest Model** | 12.68% | +217% |
| **Logistic Regression Engine** | **14.08%** | **+252% (Beats Majority)** |

---

## Troubleshooting

### Issue: `JAVA_HOME is not set` Runtime Exceptions

* **Reason**: Apache Spark cannot locate your local Java executable framework path.
* **Fix**: Explicitly bind your Java environment variables inside your execution script or terminal context:
```python
import os
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-11" # Adjust path to your machine setup

```



### Issue: Streamlit errors loading `model_metadata.json`

* **Reason**: The predictive UI was booted before the Spark pipeline notebook finalized data generation.
* **Fix**: Re-run the final compilation block inside `MIMIC_IV.ipynb` to guarantee the JSON parameter files are properly generated in your project folder, then refresh the dashboard interface.
