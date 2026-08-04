# Deep Learning Breast Cancer Detection using Transfer Learning and CNN-Based Mammogram Classification

A Final Year Project submitted in partial fulfilment of the requirements for the **BSc (Hons) Computer Science** programme at the **University of London (International Programmes)**.

**Author:** Rhithika S

---

> **Repository Status**
>
> This repository accompanies the implementation of my Final Year Project. It contains the experimental notebook and feature prototype developed as part of the study. The accompanying dissertation should be regarded as the primary reference for the complete methodology, experimental design, evaluation and discussion of findings.

---

# Project Overview

This repository contains the implementation developed for my Final Year Project, which investigates the use of deep learning for benign-versus-malignant mammographic lesion classification using convolutional neural networks (CNNs).

The study systematically compares a baseline CNN trained from scratch with transfer learning models based on **VGG16** and **ResNet50**, evaluating two transfer learning strategies: **Frozen Feature Extraction** and **Partial Fine-Tuning**. The objective is to determine how different transfer learning strategies influence classification performance when applied under a consistent experimental framework.

Beyond classification performance, the study also evaluates model interpretability using **Gradient-weighted Class Activation Mapping (Grad-CAM)** to examine whether the regions highlighted by each model correspond to clinically relevant lesion areas. Following the main comparative evaluation, the best-performing configuration from each architecture is further analysed across **mass lesions** and **calcification lesions** to investigate whether performance differs between mammographic abnormality types.

A lightweight **Streamlit-based web prototype** was also developed to demonstrate comparative inference using all trained models through a single interface.

---

# Repository Guide

This repository accompanies the Final Year Project dissertation.

The primary experimental implementation is contained in:

- `FYP_DL_Study.ipynb`

The Streamlit prototype demonstrating comparative inference is located in:

- `prototype/`

The dissertation should be referred to for the complete methodology, experimental design, evaluation, results and discussion of the study.

---

# Research Objectives

This project aims to:

- Develop a baseline CNN for benign-versus-malignant mammographic lesion classification.
- Develop VGG16 and ResNet50 transfer learning models using Frozen Feature Extraction and Partial Fine-Tuning.
- Compare all developed models using multiple evaluation metrics, including Accuracy, Precision, Recall, F1-score and ROC-AUC.
- Evaluate model interpretability using Grad-CAM visualisations and compare highlighted regions against expert-provided ROI annotations.
- Analyse the best-performing CNN, VGG16 and ResNet50 configurations across mass lesions and calcification lesions to investigate performance differences between abnormality types.
- Assess the feasibility of external validation using the INBreast dataset.

---

# Dataset

The experiments were conducted using the **Kaggle distribution of the CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** dataset.

The dataset provides:

- JPEG mammographic lesion images
- Pathology-confirmed lesion labels
- Expert-provided ROI annotations
- Organised metadata files
- Cropped lesion images for lesion-level classification

The study performs binary classification by grouping the original pathology labels into:

**Benign**

- BENIGN
- BENIGN_WITHOUT_CALLBACK

**Malignant**

- MALIGNANT

The official CBIS-DDSM training and test partitions are preserved. The official training set is further divided into training and validation subsets using a stratified 80:20 split, while the official test set is reserved exclusively for final model evaluation.

---

# Models Evaluated

Five model configurations are implemented and compared throughout the study.

| Architecture | Transfer Learning Strategy |
|--------------|----------------------------|
| Baseline CNN | Trained from Scratch |
| VGG16 | Frozen Feature Extraction |
| VGG16 | Partial Fine-Tuning |
| ResNet50 | Frozen Feature Extraction |
| ResNet50 | Partial Fine-Tuning |

All transfer learning models utilise ImageNet pretrained weights while maintaining a common classification head and consistent training configuration to ensure a fair comparison between architectures.

---

# Image Processing Pipeline

The preprocessing pipeline includes:

- Image resizing (224 × 224)
- Architecture-specific preprocessing
- Data augmentation
  - Random rotation
  - Random translation
  - Random zoom
- Class weighting
- TensorFlow Dataset pipeline
- Mini-batch training
- Dataset prefetching

---

# Evaluation Strategy

The developed models are evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional evaluation includes:

- Grad-CAM interpretability analysis
- Comparison against expert ROI annotations
- Abnormality-type analysis
  - Mass lesions
  - Calcification lesions
- External validation using the INBreast dataset (where feasible)

---

# Feature Prototype

A lightweight web-based prototype was developed using **Streamlit** to demonstrate the inference capabilities of the trained models.

Current functionality includes:

- Upload cropped mammographic lesion images
- Comparative inference using all five trained models
- Majority prediction
- Model agreement summary
- Comparative prediction table
- Placeholder for Grad-CAM visualisation

The prototype demonstrates the comparative inference workflow only. Model training is performed separately within the experimental notebook.

---

# Technologies Used

## Programming Language

- Python

## Deep Learning Framework

- TensorFlow
- Keras

## Data Processing

- NumPy
- Pandas

## Image Processing

- Pillow

## Web Prototype

- Streamlit

## Development Environment

- Jupyter Notebook

---

# Repository Structure

```text
.
├── FYP_DL_Study.ipynb        # Main experimental notebook
├── prototype/                # Streamlit prototype
└── README.md
```

---

# Dataset Availability

The dataset is not included in this repository due to file size and distribution constraints.

Experiments were conducted using the **Kaggle distribution of the CBIS-DDSM dataset**, which provides organised JPEG images and accompanying metadata suitable for deep learning workflows.

---

# Disclaimer

This project was developed for academic research purposes as part of a university Final Year Project.

The developed models are intended solely for research and educational purposes and are **not** designed for clinical diagnosis or medical decision-making.

---

# Author

**Rhithika S**

BSc (Hons) Computer Science

University of London
