# Deep Learning Breast Cancer Detection using Transfer Learning and CNN-Based Mammogram Classification: A Comparative Study

Final Year Project (CM3070) for the **BSc Computer Science** programme, **University of London**.

**Author:** Rhithika S · **Supervisor:** Sarita Singh

> This repository accompanies the project report. The report is the primary reference for the methodology, results and discussion; this README explains what the code does and how to run it.

> **Research use only.** The models and prototype are not clinically validated and must not be used for diagnosis or medical decision-making.

---

## Project overview

The project compares transfer learning strategies for **benign-versus-malignant classification of mammographic lesions**, and tests whether **Grad-CAM** explanations align with expert-annotated lesion regions.

Seven configurations are trained on CBIS-DDSM lesion crops under one controlled protocol:

| Architecture | Strategy |
| --- | --- |
| Baseline CNN | Trained from scratch (reference) |
| VGG16 | Frozen feature extraction |
| VGG16 | Partial fine-tuning (block5) |
| ResNet50 | Frozen feature extraction |
| ResNet50 | Partial fine-tuning (conv5 stage) |
| EfficientNetB0 | Frozen feature extraction |
| EfficientNetB0 | Partial fine-tuning (block7 + top convolution) |

### Main findings

- Partial fine-tuning outperformed frozen feature extraction for every architecture (significant on DeLong's test after Holm correction).
- After fine-tuning, the three architectures did not differ significantly: how much a network is adapted mattered more than which network was chosen.
- Best configuration: **ResNet50 (Partial FT), test AUC 0.757** (95% CI 0.721–0.792), against 0.630 for the from-scratch baseline.
- Grad-CAM heatmaps of the transfer models pass a weight-randomisation sanity check, but localisation could not be distinguished from a trivial centre-point reference on centred lesion crops.
- On INbreast, the higher external AUC was explained by case mix and lesion type rather than better generalisation.

---

## Repository contents

```text
.
├── CM3070_breastCancerDetection_Main_Notebook1.ipynb              # Main study (CBIS-DDSM)
├── CM3070_BreastCancerDetection_External_ValidationNotebook2_.ipynb  # External validation (INbreast)
├── prototype/                                                     # Streamlit prototype (see prototype/README.md)
└── README.md
```

### Notebook 1: Main study (CBIS-DDSM)

| Section | Content |
| --- | --- |
| 1 | Data exploration |
| 2 | Preprocessing and patient-level train/validation/test split |
| 3 | Dataset preparation (tf.data pipeline, CLAHE, augmentation) |
| 4 | Model builders and shared hyperparameter search (10 trials per configuration) |
| 5 | Multi-seed retraining (3 seeds) and test evaluation |
| 6 | Evaluation, model selection and statistical testing |
| 7 | Grad-CAM analysis (IoU, pointing game, weight-randomisation check) |
| 8 | Mass vs calcification analysis |

### Notebook 2: External validation (INbreast)

Loads the selected models and settings exported by Notebook 1, prepares INbreast lesion crops from DICOM and XML annotations, and evaluates the three-seed ensembles with patient-level bootstrap confidence intervals, shortcut checks and a case-mix analysis. The mass-only subset is the primary external result.

---

## Methodology summary

- **Data:** CBIS-DDSM cropped lesions (masses and calcifications); `BENIGN_WITHOUT_CALLBACK` merged into benign.
- **Splitting:** patient-level; patients appearing in both the official training and test sets are removed from training; validation is split from training by patient (20%).
- **Preprocessing:** resize to 224 × 224, CLAHE (clip limit 2.0, 8 × 8 tiles), architecture-specific normalisation; augmentation (rotation, translation, zoom, horizontal flip) on training data only; balanced class weights.
- **Tuning:** 10 shared trials per configuration, selected on validation AUC; lower learning-rate range and frozen BatchNorm for fine-tuning.
- **Final training:** each winning setting retrained with 3 seeds; Youden's J thresholds chosen on validation; test set used once.
- **Statistics:** bootstrap 95% CIs on five metrics; DeLong's and McNemar's tests on 10 pre-specified comparisons, Holm-corrected.
- **Explainability:** logit-based, seed-averaged Grad-CAM; IoU and pointing game against expert ROIs with all-ones and centre-point references; weight-randomisation sanity check.
- **External validation:** INbreast with a BI-RADS proxy for ground truth (BI-RADS 2 benign; 4c/5/6 malignant).

---

## Streamlit prototype

The prototype runs all seven models on a single lesion crop and shows:

- the preprocessed image the models receive
- per-model predictions at their own validation-selected thresholds, with the majority prediction and model agreement
- Grad-CAM overlays for the four selected models, with IoU against expert ROIs for demo cases
- an occlusion-sensitivity map for the baseline as a gradient-free cross-check
- inference and Grad-CAM latency

Setup, verification results and limitations are in [`prototype/README.md`](prototype/README.md).

### Model files

The trained checkpoints and model artefacts (`*.keras`, `thresholds.pkl`, `calibrators.pkl`, `prototype_config.json`) are not stored in this repository because of GitHub's file size limits. Download them from [this Google Drive folder](https://drive.google.com/drive/folders/1gmivgWHeVV1eL_7NgNvXzMGrO1yvPF_h?usp=sharing) and place them in `prototype/models/`.

---

## Running the notebooks

Both notebooks were developed and run on **Google Colab** with an NVIDIA T4 GPU (Python 3.13, TensorFlow 2.20, Keras 3.13). They expect:

- the CBIS-DDSM Kaggle JPEG release (Notebook 1), downloaded via the Kaggle API
- the INbreast Kaggle mirror (Notebook 2)
- Google Drive mounted for persistent logs, checkpoints and predictions

Training cells skip completed runs, so an interrupted Colab session can resume where it stopped.

---

## Datasets

The datasets are not included in this repository.

- **CBIS-DDSM:** published by The Cancer Imaging Archive under the Creative Commons Attribution 3.0 licence. Sawyer Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). *Curated Breast Imaging Subset of DDSM* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY. Obtained via the Kaggle mirror: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- **INbreast:** Moreira, I. C. et al. (2012). INbreast: Toward a full-field digital mammographic database. *Academic Radiology, 19*(2), 236–248. Obtained via the Kaggle mirror: https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset. Used for non-commercial research only.

---

## Technologies

Python · TensorFlow / Keras · scikit-learn · SciPy · statsmodels · NumPy · pandas · OpenCV · Pillow · pydicom · scikit-image · Matplotlib · Streamlit · Google Colab

---

## Disclaimer

This project was developed for academic research as part of a university Final Year Project. The models are intended for research and educational purposes only and are **not** designed for clinical diagnosis or medical decision-making.

## Author

**Rhithika S**, BSc Computer Science, University of London
