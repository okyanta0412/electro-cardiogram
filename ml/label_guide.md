# Single-lead bedside monitor photo label guide (prototype)

This guide defines one mutually exclusive label for each photo analyzed as one ECG trace: Lead II. If a photo shows standard I/II/III rows, use only the middle Lead II row; if it shows one row, its label must be confirmed as Lead II. Other layouts are outside this prototype's input. This is not a 3x4 twelve-lead paper ECG task. Labels must come from a trusted dataset annotation or qualified human review; do not infer them from this prototype's prediction.

## Unit of classification

One monitor photo is one classification example and receives exactly one label. The complete Lead II trace and enough consecutive beats must be visible. A single isolated beat is not sufficient evidence for an AF label. If multiple target findings are present, the example is mixed and is excluded from this first single-label dataset rather than forcing a label choice. Record the monitor sweep speed and gain when available.

## Labels

| Label | Assign when |
| --- | --- |
| `NORMAL` | The image is readable and its reference annotation indicates none of the target findings (AF, PVC, or ST-T change). |
| `AF` | The source annotation or qualified reviewer confirms atrial fibrillation, and the image contains enough consecutive beats to represent the rhythm. |
| `PVC` | The source annotation or qualified reviewer confirms a ventricular premature beat in the target crop. |
| `STTC` | The source annotation or qualified reviewer confirms a target ST/T change in the shown lead(s). |
| `UNREADABLE` | The image has no identifiable Lead II trace, is too blurry/cropped/noisy to assess, or lacks enough context for its proposed label. |

## Exclusion and review

- Exclude images with two or more target findings from this initial single-label set; do not arbitrarily prioritize AF over PVC or STTC.
- Keep the patient identifier only in a private split manifest; never publish identifying patient information.
- Split train/validation/test by patient, not by image, to avoid near-duplicate patient records leaking across splits.
- Store the label, source annotation/reviewer, source record, patient split key, and image path in a CSV manifest.
- Have a qualified reviewer verify labels and dataset permissions before training or sharing results.

## Limitations

These are dataset labels for a research prototype, not clinical diagnoses. The current PM-ECG-ID paired samples are twelve-lead paper ECG photos and are not target data for this single-lead monitor task. No paired single-lead monitor photos and signals have been collected yet. The current MIT-BIH and PTB-XL signal models are not trained on these photos, and the browser page does not perform image classification.
