# ECG photo label guide (prototype)

This guide defines one mutually exclusive label for each ECG photo/crop used by a future image-classification model. Labels must come from a trusted dataset annotation or qualified human review; do not infer labels from this prototype's prediction.

## Unit of classification

One image is one classification example and receives exactly one label. Use a crop/strip that contains enough context for its target label. A single isolated beat is not sufficient evidence for an AF label. If multiple target findings are present, the example is mixed and is excluded from this first single-label dataset rather than forcing a label choice.

## Labels

| Label | Assign when |
| --- | --- |
| `NORMAL` | The image is readable and its reference annotation indicates none of the target findings (AF, PVC, or ST-T change). |
| `AF` | The source annotation or qualified reviewer confirms atrial fibrillation, and the image contains enough consecutive beats to represent the rhythm. |
| `PVC` | The source annotation or qualified reviewer confirms a ventricular premature beat in the target crop. |
| `STTC` | The source annotation or qualified reviewer confirms a target ST/T change in the shown lead(s). |
| `UNREADABLE` | The image is not an ECG, is too blurry/cropped/noisy to assess, or lacks enough context for its proposed label. |

## Exclusion and review

- Exclude images with two or more target findings from this initial single-label set; do not arbitrarily prioritize AF over PVC or STTC.
- Keep the patient identifier only in a private split manifest; never publish identifying patient information.
- Split train/validation/test by patient, not by image, to avoid near-duplicate patient records leaking across splits.
- Store the label, source annotation/reviewer, source record, patient split key, and image path in a CSV manifest.
- Have a qualified reviewer verify labels and dataset permissions before training or sharing results.

## Limitations

These are dataset labels for a research prototype, not clinical diagnoses. The current MIT-BIH and PTB-XL signal models are not trained on photos and do not implement these photo labels. The current browser page does not perform image classification.
