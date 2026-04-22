"""
prepare_breakhis.py

Downloads the BreakHis dataset, organizes it by magnification,
and splits it into train/val/test by patient ID (to avoid data leakage).

Usage:
    python prepare_breakhis.py --magnification 400X --output /workspace/data/breakhis
    python prepare_breakhis.py --magnification 40X  --output ./data/breakhis

Magnification options: 40X, 100X, 200X, 400X
"""

import os
import re
import shutil
import argparse
import zipfile
import urllib.request
import ssl
from pathlib import Path
from collections import defaultdict
import random

# ── Constants ─────────────────────────────────────────────────────────────────

DOWNLOAD_URL = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"

# All 8 classes: 4 benign, 4 malignant
BENIGN_CLASSES    = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
MALIGNANT_CLASSES = ["ductal_carcinoma", "lobular_carcinoma",
                     "mucinous_carcinoma", "papillary_carcinoma"]
ALL_CLASSES       = BENIGN_CLASSES + MALIGNANT_CLASSES

# Folder name patterns inside the BreakHis archive
# e.g. SOB_B_A-14-22549AB-40-001.png
#       ↑   ↑ ↑   ↑         ↑
#       SOB B/M class  patient  magnification
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# Patient ID is encoded in the filename: SOB_{B/M}_{class}-{patient}-{mag}-{seq}.png
PATIENT_RE = re.compile(r"SOB_[BM]_[^-]+-(\d+-\d+[A-Z]*)-\d+X-\d+\.png", re.IGNORECASE)

# Train / val / test split ratios (by patient)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> None:
    """Download a file with a progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  → {dest}")

    # Disable SSL verification (the UFPR server uses a self-signed cert)
    ctx = ssl._create_unverified_context()

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb  = downloaded / 1024 / 1024
            print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract .tar.gz archive."""
    print(f"Extracting {archive.name} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(dest))
    print("  Extraction complete.")


def find_images(root: Path, magnification: str) -> dict[str, list[Path]]:
    """
    Walk the extracted archive and collect all images for the given
    magnification, grouped by class name.

    Returns:
        { class_name: [Path, ...], ... }
    """
    # Map archive folder names → our canonical class names
    folder_to_class = {
        "A":   "adenosis",
        "F":   "fibroadenoma",
        "PT":  "phyllodes_tumor",
        "TA":  "tubular_adenoma",
        "DC":  "ductal_carcinoma",
        "LC":  "lobular_carcinoma",
        "MC":  "mucinous_carcinoma",
        "PC":  "papillary_carcinoma",
    }

    images_by_class = defaultdict(list)

    for img_path in root.rglob(f"*{magnification}*.png"):
        # Infer class from parent folder name (e.g. .../SOB/benign/adenosis/40X/...)
        for part in img_path.parts:
            part_upper = part.upper()
            if part_upper in folder_to_class:
                class_name = folder_to_class[part_upper]
                images_by_class[class_name].append(img_path)
                break
        else:
            # Fallback: infer from filename prefix
            fname = img_path.stem.upper()
            for abbrev, class_name in folder_to_class.items():
                if f"_B_{abbrev}-" in fname or f"_M_{abbrev}-" in fname:
                    images_by_class[class_name].append(img_path)
                    break

    return dict(images_by_class)


def extract_patient_id(img_path: Path) -> str:
    """
    Extract patient ID from filename.
    e.g. SOB_B_A-14-22549AB-40-001.png → "14-22549AB"
    Falls back to filename stem if pattern not matched.
    """
    m = PATIENT_RE.match(img_path.name)
    if m:
        return m.group(1)
    # Fallback: use the third dash-separated token
    parts = img_path.stem.split("-")
    return "-".join(parts[1:3]) if len(parts) >= 3 else img_path.stem


def split_patients(
    patient_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """
    Split a list of patient IDs into train / val / test sets.
    Splitting by patient prevents data leakage (same patient's images
    appearing in both train and test).
    """
    ids = sorted(set(patient_ids))   # sort for reproducibility
    random.seed(seed)
    random.shuffle(ids)

    n       = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train:n_train + n_val])
    test_ids  = set(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def copy_images(
    images:    list[Path],
    split:     str,
    class_name: str,
    output_root: Path,
) -> int:
    """Copy images into output_root/{split}/{class_name}/."""
    dest_dir = output_root / split / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        shutil.copy2(src, dest_dir / src.name)
    return len(images)


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare_breakhis(
    magnification: str,
    output_dir:    Path,
    download_dir:  Path,
    keep_archive:  bool = False,
    seed:          int  = 42,
) -> None:

    assert magnification in MAGNIFICATIONS, \
        f"magnification must be one of {MAGNIFICATIONS}, got '{magnification}'"

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ────────────────────────────────────────────────────────────
    archive_path = download_dir / "BreaKHis_v1.tar.gz"
    if archive_path.exists():
        print(f"Archive already exists at {archive_path} — skipping download.")
    else:
        download_file(DOWNLOAD_URL, archive_path)

    # ── 2. Extract ─────────────────────────────────────────────────────────────
    extract_dir = download_dir / "BreaKHis_v1"
    if extract_dir.exists():
        print(f"Already extracted at {extract_dir} — skipping extraction.")
    else:
        extract_archive(archive_path, download_dir)

    # ── 3. Collect images by class ─────────────────────────────────────────────
    print(f"\nCollecting {magnification} images...")
    images_by_class = find_images(extract_dir, magnification)

    if not images_by_class:
        raise RuntimeError(
            f"No images found for magnification {magnification} under {extract_dir}.\n"
            f"Check that the archive extracted correctly."
        )

    for cls in ALL_CLASSES:
        n = len(images_by_class.get(cls, []))
        print(f"  {cls:<25} {n:>5} images")

    # ── 4. Split by patient ID ─────────────────────────────────────────────────
    print(f"\nSplitting by patient ID "
          f"(train={TRAIN_RATIO:.0%} / val={VAL_RATIO:.0%} / test={TEST_RATIO:.0%})...")

    split_counts = defaultdict(lambda: defaultdict(int))

    for class_name in ALL_CLASSES:
        images = images_by_class.get(class_name, [])
        if not images:
            print(f"  WARNING: no images found for class '{class_name}' — skipping.")
            continue

        # Group images by patient
        patients = defaultdict(list)
        for img in images:
            pid = extract_patient_id(img)
            patients[pid].append(img)

        # Split patient IDs
        train_ids, val_ids, test_ids = split_patients(
            list(patients.keys()), TRAIN_RATIO, VAL_RATIO, seed=seed
        )

        print(f"  {class_name:<25} "
              f"{len(train_ids)} train patients / "
              f"{len(val_ids)} val patients / "
              f"{len(test_ids)} test patients")

        # Copy images into output directory
        for pid, imgs in patients.items():
            if pid in train_ids:
                split = "train"
            elif pid in val_ids:
                split = "val"
            else:
                split = "test"
            n = copy_images(imgs, split, class_name, output_dir)
            split_counts[split][class_name] += n

    # ── 5. Print summary ───────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"{'Class':<25} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"{'─' * 55}")
    total_train = total_val = total_test = 0
    for cls in ALL_CLASSES:
        tr = split_counts["train"].get(cls, 0)
        va = split_counts["val"].get(cls, 0)
        te = split_counts["test"].get(cls, 0)
        total_train += tr
        total_val   += va
        total_test  += te
        print(f"  {cls:<23} {tr:>8} {va:>8} {te:>8}")
    print(f"{'─' * 55}")
    print(f"  {'TOTAL':<23} {total_train:>8} {total_val:>8} {total_test:>8}")
    print(f"{'─' * 55}")
    print(f"\nOutput written to: {output_dir}")
    print(f"  {output_dir}/train/{{class_name}}/*.png")
    print(f"  {output_dir}/val/{{class_name}}/*.png")
    print(f"  {output_dir}/test/{{class_name}}/*.png")

    # ── 6. Cleanup ─────────────────────────────────────────────────────────────
    if not keep_archive:
        print(f"\nRemoving archive and extracted files to save disk space...")
        shutil.rmtree(extract_dir, ignore_errors=True)
        archive_path.unlink(missing_ok=True)
        print("  Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare the BreakHis dataset."
    )
    parser.add_argument(
        "--magnification",
        type=str,
        default="400X",
        choices=MAGNIFICATIONS,
        help="Magnification level to use (default: 400X)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/data/breakhis"),
        help="Output directory for train/val/test splits (default: /workspace/data/breakhis)",
    )
    parser.add_argument(
        "--download_dir",
        type=Path,
        default=Path("/workspace/data/breakhis_raw"),
        help="Directory to store the downloaded archive (default: /workspace/data/breakhis_raw)",
    )
    parser.add_argument(
        "--keep_archive",
        action="store_true",
        help="Keep the raw archive and extracted files after processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args()

    prepare_breakhis(
        magnification=args.magnification,
        output_dir=args.output,
        download_dir=args.download_dir,
        keep_archive=args.keep_archive,
        seed=args.seed,
    )

