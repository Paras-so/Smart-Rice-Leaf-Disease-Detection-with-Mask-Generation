"""Rebuild training data for the saved experiments from the original images."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Dataset" / "Images"
CLEAN = ROOT / "cleaned dataset"
SPLIT = ROOT / "split_dataset"
MANIFEST = ROOT / "results" / "provenance" / "split_manifest.csv"


def verify(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def copy_image(source, target, digest):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    verify(target, digest)


def rebuild_images():
    if not RAW.is_dir():
        raise FileNotFoundError(f"Place the original class folders under {RAW}")
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        verify(RAW / row["path"], row["sha256"])
    split_manifest = SPLIT / "manifest.csv"
    if split_manifest.exists() and split_manifest.read_bytes() != MANIFEST.read_bytes():
        raise ValueError(f"Existing manifest differs from the saved split: {split_manifest}")
    for row in rows:
        clean = CLEAN / row["path"]
        copy_image(RAW / row["path"], clean, row["sha256"])
        copy_image(clean, SPLIT / row["split"] / row["path"], row["sha256"])
    shutil.copyfile(MANIFEST, split_manifest)
    print(f"Rebuilt {len(rows)} deduplicated images and their original splits.")


def run_cells(notebook, indices, namespace):
    path = ROOT / notebook
    cells = json.loads(path.read_text(encoding="utf-8-sig"))["cells"]
    for index in indices:
        source = "".join(cells[index]["source"])
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-only", action="store_true", help="Build only cleaned and split images.")
    parser.add_argument("--unet", action="store_true", help="Also generate U-Net draft masks.")
    args = parser.parse_args()
    if args.images_only and args.unet:
        parser.error("--unet requires augmented images and cannot be combined with --images-only")
    os.chdir(ROOT)
    rebuild_images()
    if args.images_only:
        return

    import pandas as pd

    manifest = pd.read_csv(MANIFEST)
    augmentation = {
        "train": manifest.loc[manifest["split"].eq("train")],
        "SOURCE": CLEAN,
        "AUG_DIR": ROOT / "augmented_train",
    }
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from PIL import Image, ImageEnhance, ImageOps

    augmentation.update(
        Path=Path, json=json, pd=pd, np=np, Image=Image, ImageOps=ImageOps,
        ImageEnhance=ImageEnhance, ThreadPoolExecutor=ThreadPoolExecutor,
    )
    run_cells("data_split_and_augmentation.ipynb", [7], augmentation)
    if args.unet:
        annotation = {}
        run_cells("pixel_level_disease_annotation.ipynb", [2, *range(4, 9), *range(14, 26)], annotation)
        annotation["run_full_annotation"]()


if __name__ == "__main__":
    main()
