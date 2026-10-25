# ECG beat image classification prototype

This research prototype downloads the MIT-BIH Arrhythmia Database, renders annotated beats as ECG-paper-style PNG images, and trains a HOG + linear SVM classifier.

It classifies only two annotated beat symbols:

- `N`: normal beat
- `V`: premature ventricular contraction (PVC)

Other annotation symbols are excluded. This is not a classifier for atrial fibrillation, ST changes, full ECG strips, or photographs of paper ECGs. It is not for clinical use.

## Environment

Use Python 3.10 or newer. From this directory:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
```

The script downloads four MIT-BIH recordings (about 8 MB) to `data/raw/mitdb/`. It trains on records 105 and 119, and evaluates on records 106 and 108. The script reports held-out metrics and saves a model plus a few rendered example images under `models/`.

This small split is only a pipeline smoke test, not a reliable estimate of model performance.

After training, run the inference CLI on one generated 160x96 beat image:

```sh
python predict_beat_image.py models/sample_images/NORMAL.png
python predict_beat_image.py models/sample_images/PVC.png
```

The output includes an uncalibrated SVM decision margin, not a probability or clinical confidence. The CLI deliberately rejects full ECG photographs and strips.
The database contains digitized two-channel signals and expert-reviewed beat annotations, not paper photographs. Synthetic plot images are useful for testing the training pipeline, but performance on actual camera photos is unknown and must not be inferred from these results. A later photo classifier needs labeled ECG photographs and image preprocessing that isolates the relevant waveforms.

## Data source and citation

Moody GB, Mark RG. MIT-BIH Arrhythmia Database, version 1.0.0. PhysioNet. https://doi.org/10.13026/C2F305

The database is distributed under the Open Data Commons Attribution License v1.0. Review the license and cite the database in any report or presentation: https://physionet.org/content/mitdb/1.0.0/

## Separate AFIB/STTC numerical-signal experiment

`train_af_st_models.py` is a separate experiment using PTB-XL 1.0.3 12-lead numerical signals (10 seconds at 100 Hz). It creates two binary `StandardScaler` + `LinearSVC` models: `AFIB` versus not-AFIB, and `STTC` versus not-STTC. STTC membership comes from PTB-XL's official SCP diagnostic-class mapping. The official stratified folds 1-8 are used for training and folds 9-10 for held-out evaluation.

Run it from this directory after installing the requirements:

```sh
python train_af_st_models.py
python predict_ptbxl_signal.py data/raw/ptbxl_cache/<ecg-id>.npz
```

The training script downloads PTB-XL metadata and only the selected low-resolution records; cached signals are stored under `data/raw/ptbxl_cache/`. It saves separate AFIB/STTC models and metrics under `models/`. This experiment uses numerical signals, not rendered beat plots or photographs, and does not connect to the browser upload UI. Its small sampled subset and uncalibrated SVM margins are for a pipeline prototype only, not for clinical decisions.

Dataset: Wagner P, et al. PTB-XL, a large publicly available electrocardiography dataset. Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6. PTB-XL 1.0.3 is distributed under CC BY 4.0; retain attribution and review the license before redistribution: https://physionet.org/content/ptb-xl/1.0.3/ and https://physionet.org/content/ptb-xl/view-license/1.0.3/
