# Breast Cancer Detection Prototype

A Streamlit research demonstrator for the CM3070 final-year project *Deep Learning Breast Cancer Detection using Transfer Learning and CNN-Based Mammogram Classification: A Comparative Study*. It runs the trained models on a single cropped mammographic lesion image and shows their predictions alongside Grad-CAM explanations.

> **Research demonstrator only.** This application has not been clinically validated and must not be used for diagnostic or clinical decisions.

The prototype is described in Section 4.9 of the report and evaluated in Section 5.6.

## What the application does

- Accepts one of six bundled demo cases (with known ground truth) or an uploaded PNG/JPG image.
- Validates the input and shows the resized, CLAHE-enhanced image the models receive.
- Runs all seven trained configurations, each at its own decision threshold, and reports the majority prediction and model agreement.
- Shows a per-model table of predicted class, calibrated confidence and decision threshold.
- Generates Grad-CAM heatmaps for the four models selected in the report: the Baseline CNN and the partial fine-tuning configurations of VGG16, ResNet50 and EfficientNetB0.
- For demo cases, reports IoU between each heatmap and the expert ROI (requires the original dataset files, see below).
- Generates an occlusion-sensitivity map for the Baseline CNN as a gradient-free cross-check.
- Displays inference and Grad-CAM latency for each run.

The models expect **cropped lesion images**, not full mammograms. Input checks catch obviously unsuitable images but cannot confirm that an image is a lesion crop.

## Project structure

```text
FYP_Prototype/
├── app.py                          # Streamlit interface, input validation, analysis flow
├── requirements.txt                # Python dependencies
├── fit_calibrators_correct.py      # Fits the isotonic calibrators (run in the notebook environment)
├── verify_prototype_predictions.py # Legacy verification script (see "Verification")
├── pooled_export.pkl               # 50 test images + pooled pipeline predictions, for verification
├── assets/                         # Logo and interface graphics
├── demo_images/                    # Demo case 1
├── models/
│   ├── *.keras                     # Seven trained checkpoints
│   ├── thresholds.pkl              # Per-model Youden's J thresholds (this checkpoint's own validation set)
│   ├── calibrators.pkl             # Isotonic calibrators, one per configuration
│   ├── prototype_config.json       # CLAHE settings exported from the notebook
│   ├── demo_reference.xls          # Demo labels and original dataset paths (CSV format)
│   └── case_2.jpg ... case_6.jpg   # Demo cases 2 to 6
├── tests/
│   └── test_prototype_parity.py    # Prototype vs pipeline parity test
└── utils/
    ├── model_loader.py             # Loads the seven checkpoints
    ├── preprocessing.py            # Resize, CLAHE, model-specific normalisation
    ├── predictor.py                # Predictions, thresholds, majority and agreement
    ├── thresholds.py               # Loads per-model decision thresholds
    ├── calibration.py              # Applies isotonic calibration to displayed confidence
    ├── frozen_backbones.py         # Rebuilds frozen ImageNet backbones for head-only checkpoints
    ├── gradcam.py                  # Logit-based Grad-CAM and overlays
    ├── occlusion.py                # Occlusion sensitivity (32 x 32 patches)
    └── roi.py                      # Demo-case ROI alignment and IoU
```

## Models

Each configuration uses **one checkpoint**: the winning hyperparameter-search trial (report Section 5.1). The report's headline results average three training seeds per configuration, so the prototype approximates rather than exactly reproduces them (see "Verification").

| Configuration | Checkpoint | Grad-CAM |
| --- | --- | --- |
| Baseline CNN | `baseline_cnn__trial1.keras` | Yes |
| VGG16 (Frozen) | `vgg16_frozen__trial1.keras` | No |
| VGG16 (Partial Fine-Tuning) | `vgg16_partial__trial1.keras` | Yes |
| ResNet50 (Frozen) | `resnet50_frozen__trial3.keras` | No |
| ResNet50 (Partial Fine-Tuning) | `resnet50_partial__trial3.keras` | Yes |
| EfficientNetB0 (Frozen) | `effnet_frozen__trial9.keras` | No |
| EfficientNetB0 (Partial Fine-Tuning) | `effnet_partial__trial4.keras` | Yes |

Frozen checkpoints contain only the trained classification head. At inference, the app rebuilds the matching frozen ImageNet backbone (downloaded by Keras on first use, so an internet connection is needed the first time) to produce the pooled features that head expects.

## Inference and explanation flow

1. **Input.** A demo case is selected or an image uploaded.
2. **Validation.** Images smaller than 100 px on either side, with an aspect ratio above 3:1, or that fail to decode are rejected and the analysis button is disabled. Noticeably colourful images trigger a warning but are allowed.
3. **Preprocessing.** As in the experimental pipeline: resize to 224 x 224, CLAHE (clip limit 2.0, 8 x 8 tiles, read from `prototype_config.json`), then each architecture's own normalisation.
4. **Prediction.** Each model's raw malignant probability is compared with its own Youden's J threshold from `thresholds.pkl` -- computed on that exact checkpoint's own validation predictions, not a shared/pooled value, and not a fixed 0.5.
5. **Confidence.** The displayed confidence is adjusted by that model's isotonic calibrator. The class is always decided on the raw probability.
6. **Explanation.** Grad-CAM is computed on the pre-sigmoid logit at each model's last convolutional layer (`conv2d_2`, `block5_conv3`, `conv5_block3_out`, `top_conv`), using the same method as the report (Section 4.7).

Grad-CAM and occlusion maps show which regions influenced a prediction, not whether the prediction is correct. ResNet50 (Partial Fine-Tuning) produces **blank Grad-CAM maps for malignant predictions**. This is a documented finding of the study (report Section 5.4.2), not an application error, and the app flags it when it occurs.

## Calibrators

`calibrators.pkl` holds one isotonic regression per configuration. `fit_calibrators_correct.py` fits each calibrator on **that checkpoint's own validation-set predictions** (frozen configurations from the cached validation features, others from the validation dataset). The test set is never used. Because the validation set has a higher malignant rate than the test set (48.2% vs 39.2%), calibrated confidence may run slightly high on test-like data.

## Setup and launch

Use Python 3.10 to 3.13 (current TensorFlow wheels do not support Python 3.14). From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The prototype was developed with Streamlit 1.58.0 and TensorFlow 2.20. The checkpoints and model artefacts must be present in `models/`.

**Model checkpoints and artefacts:** the seven `.keras` checkpoints plus `calibrators.pkl`, `thresholds.pkl`, and `prototype_config.json` are too large for this repo (two checkpoints alone exceed GitHub's 100MB limit) and are hosted separately: [Google Drive folder](https://drive.google.com/drive/folders/1gmivgWHeVV1eL_7NgNvXzMGrO1yvPF_h?usp=sharing). Download everything from that folder into `models/` before running the app.

## Verification

`tests/test_prototype_parity.py` runs the prototype's own preprocessing and prediction code (including, for the three frozen configurations, the same rebuilt-backbone step `app.py` uses) on the 50 test images embedded in `pooled_export.pkl`, and compares its labels against the pipeline's pooled three-seed predictions for all seven configurations:

```powershell
pytest tests/ -v -s
```

Latest results, all seven configurations:

| Model | Label agreement |
| --- | --- |
| Baseline CNN | 29/50 (58%) |
| VGG16 (Frozen) | 47/50 (94%) |
| VGG16 (Partial Fine-Tuning) | 43/50 (86%) |
| ResNet50 (Frozen) | 44/50 (88%) |
| ResNet50 (Partial Fine-Tuning) | 46/50 (92%) |
| EfficientNetB0 (Frozen) | 42/50 (84%) |
| EfficientNetB0 (Partial Fine-Tuning) | 45/50 (90%) |

Disagreement is expected, because a single checkpoint cannot exactly reproduce a three-seed average, and disagreement concentrates on images whose probability sits close to that model's decision threshold. It is largest for the Baseline CNN, whose threshold (0.059) sits close to zero, making it more sensitive to small probability shifts between the deployed checkpoint and the pooled reference. The test reports every model's agreement rate but does not fail the suite on it (a `[CAUTION]` line marks anything under 90%) -- this gap is disclosed as an accepted limitation of deploying one checkpoint rather than an ensemble (report Section 5.6), not treated as a broken build.

`verify_prototype_predictions.py` is an equivalent standalone script (same embedded-image, same frozen-backbone handling) for checking parity outside of pytest, e.g. interactively.

## Demo cases and ROI comparison

The six demo cases come from the CBIS-DDSM test set. Their labels and original file paths are listed in `models/demo_reference.xls` (a CSV file despite the extension). The paths are absolute paths on the author's machine, so the IoU comparison with expert ROIs only works where the original CBIS-DDSM files exist at those paths. Elsewhere, the app shows a warning and continues without IoU.

## Data and licences

CBIS-DDSM is published by The Cancer Imaging Archive under the Creative Commons Attribution 3.0 Unported licence and the TCIA Data Usage Policy:

- Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). *Curated Breast Imaging Subset of DDSM* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
- Clark, K. et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and operating a public information repository. *Journal of Digital Imaging*, 26(6), 1045–1057.

## Limitations

- Single-checkpoint predictions approximate, but do not exactly reproduce, the report's three-seed results -- see "Verification" above for the actual per-model agreement rates.
- Displayed confidence is calibrated; the report's calibration analysis (Section 5.2.8) uses raw probabilities.
- Grad-CAM localisation could not be shown to beat trivial centre-point references on centred lesion crops (report Section 5.4.2).
- The ROI/IoU comparison for demo cases depends on `models/demo_reference.xls`'s absolute paths into the original CBIS-DDSM files on the development machine; it only works where those exact paths exist, and fails gracefully (a warning, no IoU shown) elsewhere.
- Not validated for clinical use.