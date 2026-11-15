"""Lab 3 inputs, manual ROI extraction and training-only scaling.

Import and call these functions in analysis.ipynb. Acquisition is explicit;
prepare() and verify() use saved local inputs and never access the network.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
SOURCE_DIRS = {"krakow": ROOT / "lab1/data/sentinel2/krakow",
               "bialowieza": ROOT / "lab1/data/sentinel2/bialowieza",
               "tychy": DATA / "sentinel2/tychy"}
BAD_SCL = (0, 1, 3, 7, 8, 9, 10, 11)
BANDS = ("B04", "B03", "B02")
LABELS = {0: "forest", 1: "urban", 255: "ignored/unlabelled"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def acquire_test():
    """Download only the new AOI; never refresh/modify earlier-lab inputs."""
    config = json.loads((HERE / "test_aoi.json").read_text())
    folder = SOURCE_DIRS["tychy"]
    if (folder / "metadata.json").exists():
        meta = json.loads((folder / "metadata.json").read_text())
        if meta["configuration"] != config:
            raise ValueError("Saved test configuration differs; choose a new output explicitly")
        for filename, expected in meta["sha256"].items():
            if digest(folder / filename) != expected:
                raise ValueError(f"Changed saved input: {folder / filename}")
        return meta
    sys.path.insert(0, str(ROOT / "lab1"))
    from prepare_sentinel2 import acquire
    folder.mkdir(parents=True, exist_ok=True)
    return acquire(config, config["aois"][0], DATA / "sentinel2")


def load_sources():
    """Hash-check, validate calibration/grid, and compute optical QA masks."""
    sources = {}
    previous = json.loads((ROOT / "lab2/data/manifest.json").read_text())
    for key, folder in SOURCE_DIRS.items():
        meta = json.loads((folder / "metadata.json").read_text())
        item = json.loads((folder / "source_item.json").read_text())
        if key != "tychy":
            for name, expected in previous["sha256"].items():
                path = ROOT / name
                if path.parent == folder:
                    assert digest(path) == expected, f"Changed reused input: {path}"
        for filename, expected in meta["sha256"].items():
            if digest(folder / filename) != expected:
                raise ValueError(f"Changed source: {folder / filename}")
        assert item["collection"] == "sentinel-2-c1-l2a"
        names = tuple(item["assets"][b]["eo:bands"][0]["name"] for b in ("red", "green", "blue"))
        assert names == BANDS
        with rasterio.open(folder / "rgb_reflectance.tif") as ref, \
                rasterio.open(folder / "rgb_dn.tif") as raw, \
                rasterio.open(folder / "scl.tif") as qa:
            x, dn, scl = ref.read(), raw.read(), qa.read(1)
            assert ref.shape == raw.shape == qa.shape == (512, 512)
            assert ref.crs == raw.crs == qa.crs
            assert ref.transform == raw.transform == qa.transform
            assert ref.res == (10, 10) and ref.count == raw.count == 3
            assert meta["crs"] == ref.crs.to_string()
            np.testing.assert_allclose(list(ref.transform)[:6], meta["transform"], rtol=0, atol=0)
            np.testing.assert_allclose(ref.bounds, meta["bounds"], rtol=0, atol=0)
            for index, color in enumerate(("red", "green", "blue")):
                spec = item["assets"][color]["raster:bands"][0]
                conversion = meta["conversions"][color]
                assert spec["scale"] == conversion["scale"]
                assert spec.get("offset", 0) == conversion["effective_offset"]
                np.testing.assert_allclose(x[index], dn[index].astype("float32") * spec["scale"]
                                           + spec.get("offset", 0), rtol=0, atol=1e-7)
            valid = (ref.read_masks().all(axis=0) & raw.read_masks().all(axis=0)
                     & (dn > 0).all(axis=0) & np.isfinite(x).all(axis=0)
                     & ~np.isin(scl, BAD_SCL))
            profile = ref.profile.copy()
        if key != "tychy":
            path = ROOT / f"lab2/data/sentinel2/{key}/valid_mask.tif"
            expected = previous["sha256"][relative(path)]
            assert digest(path) == expected, f"Changed lab 2 mask: {path}"
            with rasterio.open(path) as saved:
                assert saved.crs == profile["crs"] and saved.transform == profile["transform"]
                np.testing.assert_array_equal(valid, saved.read(1).astype(bool))
        expected_rgb = np.moveaxis(np.rint(255 * np.clip(x / 0.3, 0, 1) ** (1/2.2)).astype("uint8"), 0, -1)
        rgb = np.array(Image.open(folder / "rgb.png").convert("RGB"))
        np.testing.assert_array_equal(rgb, expected_rgb)
        sources[key] = dict(x=x, rgb=rgb, valid=valid, scl=scl, profile=profile,
                            meta=meta, item=item, folder=folder)
    # All grids are UTM 34N; require the independent test footprint not to intersect training.
    for key in ("krakow", "bialowieza"):
        a, b = sources[key]["meta"]["bounds"], sources["tychy"]["meta"]["bounds"]
        assert sources[key]["profile"]["crs"] == sources["tychy"]["profile"]["crs"]
        assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
        assert sources[key]["meta"]["scene_id"] != sources["tychy"]["meta"]["scene_id"]
    return sources


def write_mask(path, values, source, description):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = source["profile"].copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(path, "w", **profile) as output:
        output.write(values.astype("uint8"), 1)
        output.set_band_description(1, description)
        output.update_tags(label_encoding=json.dumps(LABELS))


def label_masks(sources, rois):
    masks = {key: {split: np.full(s["valid"].shape, 255, dtype="uint8")
                   for split in ("train", "holdout", "test")} for key, s in sources.items()}
    occupancy = {key: np.zeros(s["valid"].shape, bool) for key, s in sources.items()}
    counts = []
    for roi in rois["rois"]:
        key, split, label = roi["source"], roi["split"], roi["label"]
        assert label in (0, 1) and split in masks[key]
        assert (key == "tychy") == (split == "test")
        x0, y0, x1, y1 = roi["box_xyxy"]
        assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512
        region = np.zeros((512, 512), bool)
        region[y0:y1, x0:x1] = True
        assert not (occupancy[key] & region).any(), f"Overlapping ROI: {roi['id']}"
        occupancy[key] |= region
        selected = region & sources[key]["valid"]
        assert selected.any(), f"Empty ROI: {roi['id']}"
        masks[key][split][selected] = label
        counts.append({**roi, "valid_pixels": int(selected.sum()),
                       "excluded_qa_pixels": int(region.sum() - selected.sum())})
    buffer = rois["train_holdout_buffer_pixels"]
    for key in ("krakow", "bialowieza"):
        expanded = maximum_filter(masks[key]["train"] != 255, size=2*buffer+1, mode="constant", cval=0)
        assert not (expanded & (masks[key]["holdout"] != 255)).any()
    return masks, counts


def prepare():
    """Rebuild deterministic samples/masks and scaling from saved inputs."""
    sources = load_sources()
    rois = json.loads((HERE / "rois.json").read_text())
    masks, counts = label_masks(sources, rois)
    rng = np.random.default_rng(rois["seed"])
    rows, labels, source_ids, flat_indices = [], [], [], []
    for label in (0, 1):
        candidates = []
        for source_id, key in enumerate(("krakow", "bialowieza")):
            candidates.extend((source_id, int(i)) for i in np.flatnonzero(masks[key]["train"] == label))
        size = rois["training_samples_per_class"]
        assert len(candidates) >= size
        selection = np.sort(rng.choice(len(candidates), size=size, replace=False))
        for selected in selection:
            source_id, index = candidates[selected]
            key = ("krakow", "bialowieza")[source_id]
            rows.append(sources[key]["x"].reshape(3, -1)[:, index])
            labels.append(label)
            source_ids.append(source_id)
            flat_indices.append(index)
    x = np.asarray(rows, dtype="float64")
    y = np.asarray(labels, dtype="uint8")
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(std == 0, 1.0, std)
    z = (x - mean) / scale
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    assert np.isfinite(z).all()
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "training_samples.npz", X_raw=x, X_scaled=z, y=y,
                        source_id=np.asarray(source_ids, dtype="uint8"),
                        flat_index=np.asarray(flat_indices, dtype="int64"),
                        mean=mean, scale=scale)
    for key, source in sources.items():
        folder = DATA / "labels" / key
        write_mask(folder / "valid_mask.tif", source["valid"], source, "0 invalid, 1 valid; optical QA")
        for split, mask in masks[key].items():
            if (mask != 255).any():
                write_mask(folder / f"{split}_labels.tif", mask, source, "0 forest, 1 urban, 255 unlabelled")
    scaling = {"bands": BANDS, "units_before_scaling": "dimensionless BOA reflectance",
               "fit_on": f"exact saved {len(y)} training samples only", "ddof": 0,
               "mean": mean.tolist(), "raw_std": std.tolist(), "scale": scale.tolist(),
               "zero_std_rule": "replace exactly zero standard deviation with 1",
               "apply": "(X_raw - mean) / scale; same parameters for every image"}
    dump(DATA / "scaling.json", scaling)
    manifest = {"phase": 1, "bands": BANDS, "label_encoding": LABELS,
                "seed": rois["seed"], "training_samples_per_class": rois["training_samples_per_class"],
                "source_id_order": ["krakow", "bialowieza"],
                "buffer_pixels": rois["train_holdout_buffer_pixels"],
                "buffer_m": 10 * rois["train_holdout_buffer_pixels"],
                "roi_counts": counts, "scaling": scaling, "sources": {}}
    for key, source in sources.items():
        meta = source["meta"]
        manifest["sources"][key] = {
            "scene_id": meta["scene_id"], "acquired_utc": meta["acquired_utc"],
            "crs": meta["crs"], "bounds": meta["bounds"], "transform": meta["transform"],
            "geometry": meta["geometry"], "shape": [512, 512], "pixel_spacing_m": [10, 10],
            "valid_pixels": int(source["valid"].sum()), "conversions": meta["conversions"],
            "paths": {name: relative(source["folder"] / name) for name in
                      ("rgb_dn.tif", "rgb_reflectance.tif", "rgb.png", "scl.tif", "metadata.json", "source_item.json")},
            "counts": {split: {LABELS[label]: int((mask == label).sum()) for label in (0, 1)}
                       for split, mask in masks[key].items()}}
    tracked = [HERE / "rois.json", HERE / "test_aoi.json", HERE / "inputs.py"]
    for source in sources.values():
        tracked.extend(source["folder"] / name for name in source["meta"]["sha256"])
        tracked.extend(source["folder"] / name for name in ("metadata.json", "source_item.json"))
    tracked.extend(DATA.glob("labels/*/*.tif"))
    tracked.extend([DATA / "training_samples.npz", DATA / "scaling.json"])
    tracked.extend(ROOT / f"lab2/data/sentinel2/{key}/valid_mask.tif" for key in ("krakow", "bialowieza"))
    tracked.append(ROOT / "lab2/data/manifest.json")
    manifest["sha256"] = {relative(path): digest(path) for path in sorted(set(tracked))}
    dump(DATA / "manifest.json", manifest)
    return sources, masks, manifest


def overview(sources, masks, manifest):
    """Save source previews, georeferenced ROI overlays and enlarged ROI chips."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    colors = {0: "red", 1: "dodgerblue"}
    styles = {"train": "-", "holdout": "--", "test": ":"}
    rois = manifest["roi_counts"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (key, source) in zip(axes, sources.items()):
        ax.imshow(source["rgb"])
        for roi in rois:
            if roi["source"] != key:
                continue
            x0, y0, x1, y1 = roi["box_xyxy"]
            ax.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False,
                                  edgecolor=colors[roi["label"]], linestyle=styles[roi["split"]], linewidth=1.7))
            ax.text(x0, y0-3, roi["id"], fontsize=7, color="white",
                    bbox=dict(facecolor="black", alpha=.7, pad=1, edgecolor="none"))
        ax.set_title(f"{key.title()} | {source['meta']['acquired_utc'][:10]}\n512 × 512 | 10 m pixels")
        ax.set_xlabel("column (x)")
        ax.set_ylabel("row (y)")
    fig.legend(handles=[Patch(color=colors[0], label="forest"), Patch(color=colors[1], label="urban")],
               loc="lower center", ncol=2)
    fig.suptitle("Manual reference ROIs | solid: training; dashed: spatial holdout; dotted: independent test")
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(DATA / "labelled_overview.png", dpi=160)
    figs = [fig]
    for key, source in sources.items():
        selected = [r for r in rois if r["source"] == key]
        ncols = 4
        nrows = (len(selected)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows*3), squeeze=False)
        for ax, roi in zip(axes.flat, selected):
            x0, y0, x1, y1 = roi["box_xyxy"]
            ax.imshow(source["rgb"][y0:y1, x0:x1], interpolation="nearest")
            ax.set_title(f"{roi['id']} | {LABELS[roi['label']]} | {roi['split']}\n{x1-x0} × {y1-y0}; {roi['valid_pixels']} valid", fontsize=9)
            ax.axis("off")
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
        fig.suptitle(f"{key.title()}: manually selected ROI pixels (nearest-neighbour enlargement)")
        fig.tight_layout()
        fig.savefig(DATA / f"{key}_roi_chips.png", dpi=150)
        figs.append(fig)
    return figs


def verify():
    """Read-only offline audit of saved phase-1 arrays, masks and provenance."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert digest(ROOT / name) == expected, f"Hash mismatch: {name}"
    sources = load_sources()
    rois = json.loads((HERE / "rois.json").read_text())
    masks, counts = label_masks(sources, rois)
    assert counts == manifest["roi_counts"]
    arrays = np.load(DATA / "training_samples.npz")
    x, y, indices, ids = (arrays[name] for name in ("X_raw", "y", "flat_index", "source_id"))
    assert x.shape == (2 * rois["training_samples_per_class"], 3)
    assert len(set(zip(ids.tolist(), indices.tolist()))) == len(y)
    assert np.bincount(y).tolist() == [rois["training_samples_per_class"]]*2
    for row, label, index, source_id in zip(x, y, indices, ids):
        key = ("krakow", "bialowieza")[source_id]
        assert masks[key]["train"].flat[index] == label
        assert masks[key]["holdout"].flat[index] == 255
        np.testing.assert_array_equal(row, sources[key]["x"].reshape(3, -1)[:, index])
    # Independently recompute the seeded sample identities, not just membership.
    rng = np.random.default_rng(rois["seed"])
    expected_ids, expected_indices = [], []
    for label in (0, 1):
        candidates = [(sid, int(i)) for sid, key in enumerate(("krakow", "bialowieza"))
                      for i in np.flatnonzero(masks[key]["train"] == label)]
        for i in np.sort(rng.choice(len(candidates), size=rois["training_samples_per_class"], replace=False)):
            sid, index = candidates[i]
            expected_ids.append(sid)
            expected_indices.append(index)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(arrays["mean"], x.mean(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(arrays["scale"], np.where(x.std(axis=0) == 0, 1, x.std(axis=0)), rtol=0, atol=0)
    np.testing.assert_allclose(arrays["X_scaled"], (x-arrays["mean"])/arrays["scale"], rtol=0, atol=0)
    scaling = json.loads((DATA / "scaling.json").read_text())
    assert scaling == manifest["scaling"]
    np.testing.assert_array_equal(scaling["mean"], arrays["mean"])
    np.testing.assert_array_equal(scaling["scale"], arrays["scale"])
    for key, source in sources.items():
        expected_masks = {"valid_mask": source["valid"]}
        expected_masks.update({split+"_labels": mask for split, mask in masks[key].items() if (mask != 255).any()})
        for name, expected in expected_masks.items():
            with rasterio.open(DATA / "labels" / key / (name+".tif")) as saved:
                assert saved.crs == source["profile"]["crs"]
                assert saved.transform == source["profile"]["transform"]
                assert saved.nodata == 255
                np.testing.assert_array_equal(saved.read(1), expected)
    result = {"status": "passed", "input_and_output_hashes": len(manifest["sha256"]),
              "training_samples": len(y), "per_class": np.bincount(y).tolist(),
              "feature_count": 3, "seed": rois["seed"],
              "spatial_buffer_pixels": rois["train_holdout_buffer_pixels"],
              "independent_test": "tychy; nonoverlapping footprint and distinct scene ID",
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "rasterio", "pillow", "matplotlib", "scipy")}}
    return result
