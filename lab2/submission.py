"""Audit saved lab 2 artifacts; package files without changing notebooks.

Run verify_artifacts(ROOT) in the notebook. CLI --package only builds/checks ZIP.
Notebook structure, execution and output review are performed through the IDE.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submission_files(root=ROOT):
    """Include both labs so linked lab 1 provenance/report remains readable."""
    files = [root / name for name in ("README.md", "pyproject.toml", "uv.lock", ".gitignore")]
    for directory in ("lab1", "lab2"):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and not EXCLUDED.intersection(p.parts)
                     and not any(part.startswith(".") for part in p.relative_to(root).parts)
                     and p.suffix not in (".pyc", ".pyo"))
    return sorted(set(files))


def check_links(root, markdown, base):
    for target in re.findall(r"\]\(([^)]+)\)", markdown):
        if urlparse(target).scheme or target.startswith("#"):
            continue
        path = (base / unquote(target.split("#")[0])).resolve()
        assert path.is_relative_to(root.resolve()) and path.exists(), f"Broken local link: {target}"


def verify_artifacts(root=ROOT):
    """Verify exported files against source data, not merely file existence."""
    root = Path(root).resolve()
    lab = root / "lab2"
    manifest = read_json(lab / "data/manifest.json")
    assert set(manifest["regions"]) == {"krakow", "bialowieza"}
    for name, digest in manifest["sha256"].items():
        path = root / name
        assert not Path(name).is_absolute() and path.resolve().is_relative_to(root)
        assert sha256(path) == digest, f"Input hash mismatch: {name}"
    inputs = {}
    for region, record in manifest["regions"].items():
        values, grid = {}, None
        for sensor, fields in (("sentinel2", ("dn", "reflectance", "mask")),
                               ("sentinel1", ("linear", "db", "mask"))):
            for field in fields:
                with rasterio.open(root / record[sensor]["paths"][field]) as src:
                    current = (src.crs, src.transform, src.shape)
                    grid = grid or current
                    assert current == grid
                    values[sensor + "_" + field] = src.read(1).astype(bool) if field == "mask" else src.read()
        inputs[region] = values

    hist_dir = lab / "results/histograms"
    hist_summary = rows(hist_dir / "summary.csv")
    assert len(hist_summary) == 14
    for row in hist_summary:
        rep, region, band = row["representation"], row["region"], row["band"]
        sensor = "sentinel2" if rep.startswith("s2") else "sentinel1"
        field = "reflectance" if sensor == "sentinel2" else ("db" if rep.endswith("db") else "linear")
        index = ["B04 red", "B03 green", "B02 blue"].index(band) if sensor == "sentinel2" else ["VV", "VH"].index(band)
        x = inputs[region][sensor + "_" + field][index][inputs[region][sensor + "_mask"]].astype(float)
        data = rows(hist_dir / f"{rep}_{region}_{band.split()[0]}.csv")
        left = np.array([float(r["bin_left"]) for r in data])
        right = np.array([float(r["bin_right"]) for r in data])
        np.testing.assert_allclose(left[1:], right[:-1], atol=1e-12)
        counts = np.array([int(r["count"]) for r in data])
        actual, _ = np.histogram(x, bins=np.r_[left, right[-1]])
        np.testing.assert_array_equal(counts, actual)
        assert counts.sum() == x.size == int(row["n"])
        density = np.array([float(r["density"]) for r in data])
        np.testing.assert_allclose(density, counts / (x.size * (right - left)), rtol=1e-12)
        np.testing.assert_allclose(np.sum(density * (right - left)), 1, rtol=0, atol=1e-12)
        for column, actual_value in (("mean", x.mean()), ("std", x.std()), ("median", np.median(x))):
            np.testing.assert_allclose(float(row[column]), actual_value, rtol=1e-10, atol=1e-12)

    lsb_dir = lab / "results/lsb"
    for name, digest in read_json(lsb_dir / "run_metadata.json")["sha256"].items():
        assert sha256(root / name) == digest, f"LSB output hash mismatch: {name}"
    lsb_metrics = rows(lsb_dir / "metrics.csv")
    for region, values in inputs.items():
        with rasterio.open(lsb_dir / region / "B04_original.tif") as src:
            original = src.read(1)
        with rasterio.open(lsb_dir / region / "B04_modified.tif") as src:
            modified = src.read(1)
        np.testing.assert_array_equal(original, values["sentinel2_dn"][0])
        assert original.dtype == modified.dtype == np.uint16 and modified.shape == original.shape
        np.testing.assert_array_equal(original >> 2, modified >> 2)
        error = modified.astype(np.int32) - original.astype(np.int32)
        assert np.abs(error).max() <= 3
        valid = values["sentinel2_mask"]
        counts0, counts1 = [np.bincount(x[valid], minlength=65536) for x in (original, modified)]
        np.testing.assert_array_equal(counts0.reshape(-1, 4).sum(1), counts1.reshape(-1, 4).sum(1))
        exported = rows(lsb_dir / region / "histogram_1dn.csv")
        for key, counts in (("original", counts0), ("modified", counts1)):
            np.testing.assert_array_equal([int(r[key + "_count"]) for r in exported], counts[:len(exported)])
            assert counts[len(exported):].sum() == 0
            np.testing.assert_allclose([float(r[key + "_probability"]) for r in exported], counts[:len(exported)] / valid.sum())
        for row in (r for r in lsb_metrics if r["region"] == region):
            e = error[valid] if row["scope"] == "valid_pixels" else error.ravel()
            assert e.size == int(row["n"]) and np.count_nonzero(e) == int(row["changed_pixels"])
            np.testing.assert_allclose(float(row["mae_dn"]), np.mean(np.abs(e)), rtol=1e-12)
            np.testing.assert_allclose(float(row["rmse_dn"]), np.sqrt(np.mean(e.astype(float) ** 2)), rtol=1e-12)

    cs_dir = lab / "results/compressive_sensing"
    metrics = read_json(cs_dir / "metrics.json")
    with np.load(cs_dir / "arrays.npz", allow_pickle=False) as arrays:
        truth, measured = arrays["signal"], arrays["mask"].astype(bool)
        assert truth.shape == (500,) and measured.sum() == 50
        for row in metrics["supplied_signal"]:
            error = arrays[row["method"]] - truth
            for field, value in (("observed_residual_l2", np.linalg.norm(error[measured])),
                                 ("full_relative_l2_error", np.linalg.norm(error) / np.linalg.norm(truth)),
                                 ("unobserved_rmse", np.sqrt(np.mean(error[~measured] ** 2)))):
                np.testing.assert_allclose(row[field], value, rtol=1e-9, atol=1e-12)
        control = arrays["control_reconstruction"] - arrays["control_signal"]
        assert np.linalg.norm(control) / np.linalg.norm(arrays["control_signal"]) < 1e-7
    # Check presence and decoding of every submitted lab 2 scientific figure.
    figures = sorted((lab / "results").rglob("*.png")) + [lab / "data/input_comparison.png"]
    assert len(figures) == 12
    for path in figures:
        with Image.open(path) as image:
            assert min(image.size) >= 500
            image.verify()
    for doc in [root / "README.md", lab / "README.md", lab / "CHECKLIST.md", lab / "SUBMISSION_REVIEW.md"]:
        check_links(root, doc.read_text(), doc.parent)
    return {"status": "passed", "input_hashes": len(manifest["sha256"]),
            "histograms_verified_against_rasters": len(hist_summary),
            "lsb_regions_verified": len(inputs), "cs_methods_verified": len(metrics["supplied_signal"]),
            "figures_decoded": len(figures), "private_credentials_required_for_saved_data": False}


def build_archive(root=ROOT):
    root = Path(root).resolve()
    files = submission_files(root)
    inventory = "".join(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n" for p in files)
    destination = root / "dist/MLForSpace-lab2-submission.zip"
    destination.parent.mkdir(exist_ok=True)
    prefix = "MLForSpace-lab2/"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, prefix + path.relative_to(root).as_posix())
        archive.writestr(prefix + "SUBMISSION_FILES.sha256", inventory)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None, "ZIP CRC verification failed"
        for path in files:
            content = archive.read(prefix + path.relative_to(root).as_posix())
            assert hashlib.sha256(content).hexdigest() == sha256(path)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    if args.package:
        print(build_archive())
    else:
        print(json.dumps(verify_artifacts(), indent=2))
