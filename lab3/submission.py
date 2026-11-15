"""Audit Lab 3 saved artifacts and build a portable archive without editing notebooks.

The notebook calls verify_artifacts() and verify_relocated() in its kernel.
CLI --package only copies/checks files; notebook execution is reviewed in the IDE.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from unittest.mock import patch

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULT_FOLDERS = ("bayes", "bayes/classification", "knn", "comparison")


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submission_files(root=ROOT):
    """Include all three labs to preserve data dependencies and documentation links."""
    root = Path(root).resolve()
    files = [root / name for name in ("README.md", "pyproject.toml", "uv.lock", ".gitignore")]
    for name in ("lab1", "lab2", "lab3"):
        files.extend(path for path in (root / name).rglob("*")
                     if path.is_file() and not any(part.startswith(".") or part == "__pycache__"
                                                  for part in path.relative_to(root).parts)
                     and path.suffix not in (".pyc", ".pyo"))
    assert all(path.is_file() and not path.is_symlink() for path in files)
    return sorted(set(files))


def check_links(root):
    root = Path(root).resolve()
    checked = 0
    for doc in [root / "README.md", *(root / "lab3").rglob("*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            if urlparse(target).scheme or target.startswith("#"):
                continue
            path = (doc.parent / unquote(target.split("#")[0])).resolve()
            assert path.is_relative_to(root) and path.exists(), f"Broken local link in {doc}: {target}"
            checked += 1
    return checked


def recorded_hashes(root):
    records = read_json(root / "lab3/data/manifest.json")["sha256"].copy()
    for folder in RESULT_FOLDERS:
        base = root / "lab3/results" / folder
        assert read_json(base / "verification.json")["status"] == "passed"
        for filename, key in (("run_metadata.json", "input_hashes"), ("verification.json", "output_sha256")):
            for path, digest in read_json(base / filename)[key].items():
                assert path not in records or records[path] == digest, f"Conflicting hashes: {path}"
                records[path] = digest
    for name, expected in records.items():
        path = (root / name).resolve()
        assert not Path(name).is_absolute() and path.is_relative_to(root)
        assert sha256(path) == expected, f"Changed artifact: {name}"
    return records


def verify_artifacts(root=ROOT):
    """Recompute scientific checks from saved inputs, not just previous pass flags."""
    root = Path(root).resolve()
    lab = root / "lab3"
    hashes = recorded_hashes(root)
    inputs = load_module(lab / "inputs.py", "lab3_audit_inputs")
    bayes = load_module(lab / "bayes.py", "lab3_audit_bayes")
    knn = load_module(lab / "knn.py", "lab3_audit_knn")
    evaluation = load_module(lab / "evaluation.py", "lab3_audit_evaluation")
    assert inputs.ROOT == root
    inputs.acquire_test()  # Must verify/reuse the saved acquisition, without a download.
    inputs.verify()
    sources = inputs.load_sources()
    masks, _ = inputs.label_masks(sources, read_json(lab / "rois.json"))
    with np.load(lab / "data/training_samples.npz", allow_pickle=False) as stored:
        raw, x, y = stored["X_raw"], stored["X_scaled"], stored["y"]
        mean, scale = stored["mean"], stored["scale"]
    base = lab / "results/bayes"
    with np.load(base / "gaussian_model.npz", allow_pickle=False) as stored:
        model = {key: stored[key] for key in stored.files}
    np.testing.assert_array_equal(model["scaling_mean"], mean)
    np.testing.assert_array_equal(model["scaling_scale"], scale)
    fitted = bayes.GaussianBayes.fit(x, y, priors=model["priors"],
                                   regularization=float(model["regularization_scaled"]))
    np.testing.assert_array_equal(fitted.means, model["means_scaled"])
    np.testing.assert_array_equal(fitted.covariances, model["covariances_scaled"])
    for label in (0, 1):
        values = raw[y == label]
        np.testing.assert_allclose(model["means_raw"][label], values.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(model["covariances_raw_mle"][label], np.cov(values, rowvar=False, bias=True), atol=1e-16)
    histogram_count = 0
    for path in sorted(base.glob("*_histogram.csv")):
        data = np.genfromtxt(path, delimiter=",", names=True)
        label = int(path.name.startswith("urban_"))
        band = ("B04", "B03", "B02").index(path.name.split("_")[1])
        edges = np.r_[data["left"], data["right"][-1]]
        np.testing.assert_array_equal(data["left"][1:], data["right"][:-1])
        count, _ = np.histogram(raw[y == label, band], bins=edges)
        np.testing.assert_array_equal(data["count"], count)
        assert count.sum() == 512
        np.testing.assert_allclose(data["density"], count / (512*np.diff(edges)), atol=1e-12)
        assert abs(np.dot(data["density"], np.diff(edges))-1) < 1e-12
        if "gaussian_marginal_at_center" in data.dtype.names:
            z = (data["left"]+data["right"])/2
            variance = model["covariances_raw"][label, band, band]
            density = np.exp(-.5*(z-model["means_raw"][label, band])**2/variance)/np.sqrt(2*np.pi*variance)
            np.testing.assert_allclose(data["gaussian_marginal_at_center"], density, atol=1e-12)
        histogram_count += 1
    assert histogram_count == 9
    source = sources["tychy"]
    valid = source["valid"]
    test_raw = np.moveaxis(source["x"], 0, -1)
    test = (test_raw[valid]-mean)/scale
    paths = {"Bayes": base / "classification/tychy_bayes_classes.tif",
             **{f"kNN k={k}": lab / f"results/knn/tychy_knn_k{k}_classes.tif" for k in (1, 3, 5)}}
    maps = {}
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[0], palette[1], palette[255] = [255, 0, 0, 255], [0, 0, 255, 255], [128, 128, 128, 0]
    for name, path in paths.items():
        with rasterio.open(path) as raster:
            image = raster.read(1)
            assert raster.crs == source["profile"]["crs"] and raster.transform == source["profile"]["transform"]
            assert raster.shape == valid.shape and raster.nodata == 255 and raster.dtypes == ("uint8",)
            np.testing.assert_array_equal(raster.read_masks(1) != 0, valid)
        assert np.isin(image[valid], [0, 1]).all() and (image[~valid] == 255).all()
        color = path.with_name(path.name.replace("_classes.tif", "_colors.png"))
        np.testing.assert_array_equal(np.asarray(Image.open(color)), palette[image])
        maps[name] = image
    np.testing.assert_array_equal(fitted.predict(test), maps["Bayes"][valid])
    with np.load(base / "classification/tychy_bayes_scores.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["valid_flat_index"], np.flatnonzero(valid))
        np.testing.assert_allclose(stored["log_scores_scaled"], fitted.log_scores(test), rtol=1e-12, atol=1e-10)
    classifier = knn.BruteForceKNN(x, y)
    with np.load(lab / "results/knn/nearest_training_rows.npz", allow_pickle=False) as stored:
        indices, distances = stored["training_row"], stored["squared_distance"]
        np.testing.assert_array_equal(stored["valid_flat_index"], np.flatnonzero(valid))
    assert indices.shape == distances.shape == (int(valid.sum()), 6)
    assert np.isfinite(distances).all() and (distances >= 0).all() and (np.diff(distances, axis=1) >= 0).all()
    probes = np.linspace(0, len(test)-1, 128, dtype=int)
    expected_distances, expected_indices = classifier.kneighbors(test[probes], n_neighbors=6, batch_size=31)
    np.testing.assert_array_equal(indices[probes], expected_indices)
    np.testing.assert_array_equal(distances[probes], expected_distances)
    # Recompute all saved neighbour distances and votes in bounded blocks.
    for start in range(0, len(test), 4096):
        stop = start+4096
        delta = test[start:stop, None, :] - x[indices[start:stop]]
        np.testing.assert_allclose(distances[start:stop], np.sum(delta*delta, axis=2), rtol=1e-12, atol=1e-12)
    for k, prediction in classifier.vote(indices).items():
        np.testing.assert_array_equal(prediction, maps[f"kNN k={k}"][valid])
    reference = masks["tychy"]["test"]
    evaluate = valid & (reference != 255)
    truth = reference[evaluate]
    np.testing.assert_array_equal(np.bincount(truth, minlength=2), [2304, 256])
    reports = read_json(lab / "results/comparison/reports.json")
    summary = rows(lab / "results/comparison/metrics.csv")
    for name, image in maps.items():
        report = evaluation.metrics_from_confusion(evaluation.binary_confusion(truth, image[evaluate]))
        assert report == reports[name]
        row = next(row for row in summary if row["method"] == name)
        for field, value in report["summary"].items():
            assert float(row[field]) == value
    with np.load(lab / "results/comparison/evaluation_pixels.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["valid_reference_flat_index"], np.flatnonzero(evaluate))
        np.testing.assert_array_equal(stored["reference"], truth)
        np.testing.assert_array_equal(stored["predictions"], np.stack([image[evaluate] for image in maps.values()]))
    for row in rows(lab / "results/comparison/pairwise_changes.csv"):
        a, b = maps[row["method_from"]], maps[row["method_to"]]
        changed = valid & (a != b)
        assert int(row["changed_pixels"]) == changed.sum()
        np.testing.assert_allclose(float(row["changed_percent"]), 100*changed.sum()/valid.sum(), atol=1e-13)
        assert int(row["changed_reference_pixels"]) == (changed & evaluate).sum()
        assert int(row["reference_corrections"]) == ((a[evaluate] != truth) & (b[evaluate] == truth)).sum()
        assert int(row["reference_new_errors"]) == ((a[evaluate] == truth) & (b[evaluate] != truth)).sum()
    figures = sorted((lab / "results").rglob("*.png"))
    figures = [path for path in figures if not path.name.endswith("_colors.png")]
    figures += [lab / "data" / name for name in ("labelled_overview.png", "krakow_roi_chips.png",
                                                 "bialowieza_roi_chips.png", "tychy_roi_chips.png")]
    assert len(figures) == 15
    for path in figures:
        with Image.open(path) as image:
            assert min(image.size) >= 500
            image.verify()
    return dict(status="passed", recorded_hashes=len(hashes), training_samples=len(y),
                histogram_exports=histogram_count, classified_valid_pixels=int(valid.sum()),
                nodata_pixels=int((~valid).sum()), bayes_full_image_prediction="passed",
                knn_saved_distances_and_votes="passed", independent_knn_queries=len(probes),
                prediction_rasters=len(maps), reference_pixels=int(evaluate.sum()),
                comparison_reports=len(reports), scientific_figures_decoded=len(figures),
                local_documentation_links=check_links(root), saved_data_need_credentials=False)


def verify_relocated(root=ROOT):
    """Audit a complete portable copy with fresh module paths and no network."""
    root = Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix="mlforspace-lab3-") as temporary:
        relocated = Path(temporary) / "MLForSpace-lab3"
        for path in submission_files(root):
            target = relocated / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        with patch("socket.socket.connect", side_effect=RuntimeError("Offline relocated audit")), \
                patch("urllib.request.urlopen", side_effect=RuntimeError("Offline relocated audit")):
            result = verify_artifacts(relocated)
        return dict(status="passed", network_disabled=True, fresh_module_paths=True,
                    scope="saved-artifact/source audit; notebook execution reviewed separately through IDE",
                    files_copied=len(submission_files(root)), artifact_summary=result)


def build_archive(root=ROOT):
    """Build ZIP, verify CRC/SHA inventory, and check extracted documentation links."""
    root = Path(root).resolve()
    assert read_json(root / "lab3/results/submission_checks.json")["status"] == "passed"
    review = read_json(root / "lab3/results/notebook_review.json")
    assert review["status"] == "passed" and review["code_cell_count"] == 30
    assert review["notebook_sha256"] == sha256(root / "lab3/analysis.ipynb")
    files = submission_files(root)
    inventory = "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files)
    destination = root / "dist/MLForSpace-lab3-submission.zip"
    destination.parent.mkdir(exist_ok=True)
    prefix = "MLForSpace-lab3/"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, prefix + path.relative_to(root).as_posix())
        archive.writestr(prefix + "SUBMISSION_FILES.sha256", inventory)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)+1
        for path in files:
            assert hashlib.sha256(archive.read(prefix+path.relative_to(root).as_posix())).hexdigest() == sha256(path)
        with tempfile.TemporaryDirectory(prefix="mlforspace-zip-") as temporary:
            archive.extractall(temporary)
            extracted = Path(temporary) / "MLForSpace-lab3"
            for line in (extracted / "SUBMISSION_FILES.sha256").read_text().splitlines():
                expected, name = line.split("  ", 1)
                assert sha256(extracted / name) == expected
            links = check_links(extracted)
    digest = sha256(destination)
    destination.with_suffix(".zip.sha256").write_text(f"{digest}  {destination.name}\n")
    record = dict(status="passed", archive=destination.name, bytes=destination.stat().st_size,
                  sha256=digest, payload_files=len(files), zip_crc="passed", extracted_sha256_inventory="passed",
                  extracted_documentation_links=links, archive_prefix=prefix,
                  relocated_offline_artifact_audit="lab3/results/submission_checks.json")
    destination.with_name("MLForSpace-lab3-archive-checks.json").write_text(json.dumps(record, indent=2)+"\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="Build/check archive from already verified notebook outputs")
    args = parser.parse_args()
    print(build_archive() if args.package else json.dumps(verify_artifacts(), indent=2))
