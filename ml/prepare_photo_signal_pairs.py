from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parent
PTBXL_METADATA = ROOT / "data" / "raw" / "ptbxl_database.csv"
# This 12-lead paper-photo sample is for digitizer experiments, not bedside-monitor classification.
OUTPUT_DIR = ROOT / "data" / "paired_12lead_photo_samples"
PHOTO_ARCHIVE_URL = "https://zenodo.org/records/13617673/files/pmcardio_ecg_image_database.zip"
PHOTO_ARCHIVE_PREFIX = "final_data/visual_data/"
PTBXL_BASE = "ptb-xl/1.0.3"
PTBXL_IMAGE_METADATA = "final_data/metadata.csv"
SAMPLE_RECORDS = {
    "NORMAL": [11666, 11209, 20999],
    "AF": [12879, 9331],
    "PVC": [20159, 12629],
    "STTC": [5742, 6918, 5441],
}


def read_record(record_name: str, directory: str):
    for attempt in range(4):
        try:
            return wfdb.rdrecord(record_name, pn_dir=f"{PTBXL_BASE}/{directory}")
        except Exception:
            if attempt == 3:
                raise
            delay = 2**attempt
            print(f"Temporary waveform download error for {record_name}; retrying in {delay}s")
            time.sleep(delay)


def main() -> None:
    if not PTBXL_METADATA.exists():
        raise SystemExit(f"PTB-XL metadata not found: {PTBXL_METADATA}. Run the AF/ST preparation first.")

    ptbxl = pd.read_csv(PTBXL_METADATA).set_index("ecg_id")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    photo_dir = OUTPUT_DIR / "photos"
    signal_dir = OUTPUT_DIR / "signals"
    manifest_path = OUTPUT_DIR / "manifest.csv"
    rows = []

    with RemoteZip(PHOTO_ARCHIVE_URL) as archive:
        image_metadata = pd.read_csv(archive.open(PTBXL_IMAGE_METADATA))
        for label, ecg_ids in SAMPLE_RECORDS.items():
            for ecg_id in ecg_ids:
                source_id = f"LPAE_{ecg_id:05d}_hr"
                matches = image_metadata[
                    (image_metadata["ECG ID"] == source_id)
                    & (image_metadata["Image page"] == 0)
                    & image_metadata["Image relative path"].str.startswith("photos_iphone/")
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one iPhone photo for {source_id}, found {len(matches)}")

                image_row = matches.iloc[0]
                image_name = str(image_row["Image name"])
                image_path = f"{PHOTO_ARCHIVE_PREFIX}{image_row['Image relative path']}"
                image_output = photo_dir / label / image_name
                image_output.parent.mkdir(parents=True, exist_ok=True)
                image_output.write_bytes(archive.read(image_path))

                record = ptbxl.loc[ecg_id]
                relative_record = Path(str(record["filename_hr"]))
                signal_record = read_record(relative_record.name, relative_record.parent.as_posix())
                if signal_record.p_signal is None or signal_record.p_signal.shape != (5000, 12):
                    raise RuntimeError(f"Unexpected signal shape for ECG {ecg_id}")

                signal_output = signal_dir / f"{ecg_id}.npz"
                signal_output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    signal_output,
                    signal=signal_record.p_signal.astype(np.float32),
                    lead_names=np.asarray(signal_record.sig_name),
                    sampling_rate=np.asarray(signal_record.fs),
                )
                rows.append(
                    {
                        "label": label,
                        "ecg_id": ecg_id,
                        "patient_id": int(record["patient_id"]),
                        "photo_path": image_output.relative_to(OUTPUT_DIR).as_posix(),
                        "signal_path": signal_output.relative_to(OUTPUT_DIR).as_posix(),
                        "photo_source": "PM-ECG-ID photos_iphone",
                        "waveform_source": "PTB-XL 1.0.3 records500",
                    }
                )
                print(f"Saved {label} ECG {ecg_id}: {image_output.name} + 12-lead signal")

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} paired 12-lead paper-photo examples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
