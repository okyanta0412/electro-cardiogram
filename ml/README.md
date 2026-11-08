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

`train_af_st_models.py` is a separate experiment using PTB-XL 1.0.3 numerical signals for leads I, II, and III (10 seconds at 100 Hz). It creates two binary `StandardScaler` + `LinearSVC` models: `AFIB` versus not-AFIB, and `STTC` versus not-STTC. STTC membership comes from PTB-XL's official SCP diagnostic-class mapping. The official stratified folds 1-8 are used for training and folds 9-10 for held-out evaluation.

Run it from this directory after installing the requirements:

```sh
python train_af_st_models.py
python predict_ptbxl_signal.py data/raw/ptbxl_cache_3lead/<ecg-id>.npz
```

The training script downloads PTB-XL metadata and only the selected low-resolution records; selected leads are cached under `data/raw/ptbxl_cache_3lead/`. It can also reuse old 12-lead caches by slicing leads I, II, and III. It saves separate AFIB/STTC models and metrics as `models/af_st_3lead_signal_models.joblib` and `models/af_st_3lead_signal_metrics.json`. This experiment uses numerical signals, not monitor photos, and does not connect to the browser upload UI. Its small sampled subset and uncalibrated SVM margins are for a pipeline prototype only, not for clinical decisions.

Dataset: Wagner P, et al. PTB-XL, a large publicly available electrocardiography dataset. Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6. PTB-XL 1.0.3 is distributed under CC BY 4.0; retain attribution and review the license before redistribution: https://physionet.org/content/ptb-xl/1.0.3/ and https://physionet.org/content/ptb-xl/view-license/1.0.3/

## Small paired photo/signal sample

`prepare_photo_signal_pairs.py` uses HTTP range requests through `remotezip` to read only the PM-ECG-ID metadata and ten selected `photos_iphone` images from its 33.9 GB archive. It downloads the matching 500 Hz PTB-XL 12-lead numerical records and writes image/signal pairs plus a manifest under `data/paired_12lead_photo_samples/`. It does not download the entire image archive. These are 12-lead paper photos, not the target bedside-monitor images.

Run it from the project root with:

```sh
ml/.venv/bin/python ml/prepare_photo_signal_pairs.py
```

The sample has 3 NORMAL, 2 AF, 2 PVC and 3 STTC records. It is only a pipeline check, not enough to train or evaluate a classifier. The photo digitizer supports manual corner correction or feature-based alignment to a clean reference rendering, but glare still limits the real-photo result; it is not a usable clinical digitizer. PM-ECG-ID is distributed under GPL-3.0-or-later: https://zenodo.org/records/13617673

## Target: single-lead bedside monitor photos

The website and label guide describe photos of a bedside monitor showing one Lead II ECG trace. This workspace contains ten labeled PM-ECG-ID 12-lead paper photos paired with PTB-XL signals, plus ten generated single-Lead-II monitor renderings from those signals. The paper photos are not bedside-monitor images, and the monitor renderings are synthetic rather than camera photos. They support pipeline development but do not validate real bedside-monitor photo performance.

### One photographed synthetic-monitor extraction check

`digitize_monitor_photo.py` extracts a single high-contrast Lead II trace using grayscale Otsu thresholding with automatic bright/dark polarity selection; hue is not used. For a standard three-row I/II/III layout it extracts only the middle row. It exports x/y pixel coordinates plus a review overlay. `--plot-corners` can optionally rectify a manually selected quadrilateral in clockwise order (top-left, top-right, bottom-right, bottom-left). The no-reference path does not automatically detect screen corners or calibrate scale. Use `--reference-image` only when a clean rendering of the photographed screen is available: SIFT feature matches and RANSAC estimate a homography, and the rectified output preserves the photo's effective pixel density. The transform is rejected when there are too few reliable matches. When `--reference-image` is used, calibration values are specified in reference-image pixels and automatically converted to the registered output scale. Otherwise, `--seconds-per-pixel` and `--millivolts-per-pixel` use the input/rectified image pixels. For a paired reference signal, `--reference-npz` reports shape correlation and, with calibration, mV RMSE; it is only an evaluation input and is not used to extract the photo trace.

Example without a clean reference rendering:

```sh
ml/.venv/bin/python ml/digitize_monitor_photo.py /path/to/monitor-photo.jpg \
	--single-lead-name II \
	--seconds-per-pixel 0.00266667 \
	--millivolts-per-pixel 0.00657895 \
	--reference-npz ml/data/paired_photo_samples/signals/12879.npz \
	--out-dir /tmp/no-reference-check
```

On the one available phone photo, grayscale thresholding could not isolate a sparse trace because of glare and background variation. The earlier green-only mask extracted 3,172 points, but aligned correlation against the paired signal was only 0.310 and calibrated RMSE was 0.119 mV. This is not sufficient for classification. The pixel scales shown are specific to that photo. On one clean synthetic source waveform recolored green, red and blue, the grayscale extraction retained aligned correlations of 0.879-0.889; this confirms hue independence on that synthetic image only.

### Three-class local demo

`monitor_photo_classifier.py` trains a linear SVM on cached Lead II numerical signals and applies it to the waveform extracted from an uploaded high-contrast monitor image. ST-T changes are excluded. The remaining labels are assigned in the order PVC, AF/AFLT, then NORMAL. The current cache contributes 88 records: 48 NORMAL, 32 AF and 8 PVC. Evaluation uses four-fold stratified group splits by patient.

Train and evaluate the model:

```sh
ml/.venv/bin/python ml/monitor_photo_classifier.py
```

Run the combined local page and API from the project root:

```sh
ml/.venv/bin/python ml/serve_photo_classifier.py
```

Open `http://localhost:8765`. The server binds to loopback by default and does not save uploaded image files.

For public hosting, the repository includes a Dockerfile and Render Blueprint (`render.yaml`). Connect this GitHub repository in Render as a Blueprint and apply the `ecg-photo-classifier` web service. Render will provide the public HTTPS URL after deployment; the URL cannot be known until Render creates the service. The free instance may sleep when idle and have a cold start. Uploaded photos are sent to the host for processing; do not upload images containing personal or patient information. The application does not write uploaded image files to disk.

Current patient-grouped signal accuracy is 0.318 (balanced accuracy 0.285). This evaluation does not include photo extraction. PVC has only eight examples. The UI/API path works, but this model is not validated for photo classification and must not be used for clinical decisions.

### Standalone PAC signal detector

`train_pac_detector.py` trains a binary PAC-code-present versus PAC-code-not-recorded detector on PTB-XL Lead II numerical signals. It loads PAC-labeled records and uses the existing non-PAC cache as negatives, then evaluates with four-fold stratified patient-grouped splits. It saves `models/pac_detector.joblib` and `models/pac_detector_metrics.json`:

```sh
ml/.venv/bin/python ml/train_pac_detector.py
```

The current run used 398 PAC-coded and 149 non-PAC-coded records. Patient-grouped balanced accuracy at the default zero threshold was 0.496. The photo API applies a separate PAC-candidate margin threshold of 2.0; on out-of-fold signal predictions this produced 0.529 balanced accuracy, 0.839 precision and 0.118 recall. The PAC detector is now connected to the API as a candidate override, but photo extraction and PAC detection on photos are unvalidated. Records without a PAC code are not guaranteed to be clinically PAC-free; this is not for clinical use.

```sh
ml/.venv/bin/python ml/digitize_monitor_photo.py /path/to/monitor-photo.jpg \
	--reference-image ml/data/threelead_monitor_roundtrip/images/AF/12879.png \
	--seconds-per-pixel 0.0096153846 \
	--millivolts-per-pixel 0.0240384615 \
	--reference-npz ml/data/paired_photo_samples/signals/12879.npz \
	--out-dir /tmp/calibrated-monitor-check
```

The example is specific to a photo of the exact 12879 synthetic rendering; a matching clean rendering is required for feature-based alignment. In this run, registration found 35 RANSAC inliers with 1.86 px median reprojection error and preserved a 3.564x pixel scale. Against the matching Lead II signal, aligned normalized correlation was 0.523, calibrated RMSE was 0.0794 mV, and the extracted duration was 9.935 s. Manual corner correction on the same photo scored 0.180 under the earlier calibration. These are single-image engineering checks, not clinical or general phone-photo validation; reflections remain visible and limit extraction quality. The prototype assumes bright-green traces and standard I/II/III order for three-row images and does not recognize lead labels automatically.

### Synthetic single-Lead-II round-trip check

`monitor_roundtrip.py` selects Lead II from the ten paired PTB-XL records, draws one trace in a synthetic bedside-monitor layout with a known 10-second sweep, 25 mm/s, and 10 mm/mV gain, then extracts the pixels and compares the trace with its source signal in mV. Run it with:

```sh
ml/.venv/bin/python ml/monitor_roundtrip.py
```

This is only a code-path/quantization check. Since the same script draws and reads the monitor image using a known scale, the metrics do not measure performance on phone photos or real monitor screenshots. Validation on independent real-monitor images is still required; this prototype is not for clinical use.
