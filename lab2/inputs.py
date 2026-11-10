"""Prepare and verify the lab 2 pair; all saved paths are repository-relative.

Default: verify/reuse local inputs offline. --refresh searches and downloads S1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "lab2"
sys.path.insert(0, str(ROOT / "lab1"))
from http_range import open_asset

AOIS = ("krakow", "bialowieza")
BAD_SCL = (0, 1, 3, 7, 8, 9, 10, 11)
CATALOGUE = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
MANIFEST = HERE / "data/manifest.json"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "MLForSpace-lab2"}), timeout=90) as response:
        return json.load(response)


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_tif(path, data, reference, descriptions, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=reference.width,
                       height=reference.height, count=len(data), dtype=data.dtype,
                       crs=reference.crs, transform=reference.transform,
                       compress="deflate", nodata=nodata) as dst:
        dst.write(data)
        for index, name in enumerate(descriptions, 1):
            dst.set_band_description(index, name)


def prepare_s2(key):
    folder = ROOT / "lab1/data/sentinel2" / key
    meta = json.loads((folder / "metadata.json").read_text())
    item = json.loads((folder / "source_item.json").read_text())
    for name, sha in meta["sha256"].items():
        if digest(folder / name) != sha:
            raise ValueError(f"Changed lab 1 input: {folder / name}")
    bands = [item["assets"][b]["eo:bands"][0]["name"] for b in ("red", "green", "blue")]
    assert bands == ["B04", "B03", "B02"]
    with rasterio.open(folder / "rgb_reflectance.tif") as ref, \
            rasterio.open(folder / "rgb_dn.tif") as dn, rasterio.open(folder / "scl.tif") as scl:
        reflectance, raw, classes = ref.read(), dn.read(), scl.read(1)
        assert ref.shape == dn.shape == scl.shape == (512, 512)
        assert ref.crs == dn.crs == scl.crs
        assert ref.transform == dn.transform == scl.transform
        assert ref.res == (10, 10) and ref.count == dn.count == 3
        for i, color in enumerate(("red", "green", "blue")):
            conversion = meta["conversions"][color]
            expected = raw[i].astype("float32") * conversion["scale"] + conversion["effective_offset"]
            np.testing.assert_allclose(reflectance[i], expected, rtol=0, atol=1e-7)
        mask = (ref.read_masks().all(axis=0) & dn.read_masks().all(axis=0)
                & np.isfinite(reflectance).all(axis=0) & (raw > 0).all(axis=0)
                & ~np.isin(classes, BAD_SCL))
        mask_path = HERE / "data/sentinel2" / key / "valid_mask.tif"
        write_tif(mask_path, mask[None].astype("uint8"), ref, ["1 = valid for all three optical bands"])
        record = {"scene_id": meta["scene_id"], "acquired_utc": meta["acquired_utc"],
                  "bands": bands, "units": "dimensionless surface reflectance (BOA)",
                  "crs": ref.crs.to_string(), "shape": list(ref.shape),
                  "pixel_spacing_m": list(ref.res), "bounds": list(ref.bounds),
                  "transform": list(ref.transform)[:6], "geometry": meta["geometry"],
                  "conversions": meta["conversions"], "valid_pixels": int(mask.sum()),
                  "excluded_pixels": int(mask.size - mask.sum()), "scl_fractions": meta["scl_fractions"],
                  "paths": {name: relative(folder / filename) for name, filename in {
                      "reflectance": "rgb_reflectance.tif", "dn": "rgb_dn.tif",
                      "preview": "rgb.png", "scl": "scl.tif", "metadata": "metadata.json",
                      "source_item": "source_item.json"}.items()}}
        record["paths"]["mask"] = relative(mask_path)
    return meta, record


def candidates_for(meta):
    coords = meta["geometry"]["coordinates"][0]
    bbox = [min(p[0] for p in coords), min(p[1] for p in coords),
            max(p[0] for p in coords), max(p[1] for p in coords)]
    target = utc(meta["acquired_utc"])
    interval = "/".join((target + timedelta(days=d)).isoformat() for d in (-12, 12))
    params = {"collections": COLLECTION, "bbox": ",".join(map(str, bbox)),
              "datetime": interval, "limit": 100}
    url = CATALOGUE + "/search?" + urlencode(params)
    items = []
    while url:
        page = fetch(url)
        items.extend(page["features"])
        link = next((x for x in page.get("links", []) if x["rel"] == "next"), None)
        if link and link.get("method", "GET") != "GET":
            raise ValueError("Unsupported catalogue pagination method")
        url = link["href"] if link else None
    items = [x for x in items if all(p in x["assets"] for p in ("vv", "vh"))]
    return sorted(items, key=lambda x: (abs((utc(x["properties"]["datetime"]) - target).total_seconds()), x["id"]))


def signed_asset(asset):
    # Tokens are used only in memory; signed URLs never enter saved metadata.
    token = fetch("https://planetarycomputer.microsoft.com/api/sas/v1/token/"
                  "sentinel1euwestrtc/sentinel1-grd-rtc")["token"]
    return {**asset, "href": asset["href"] + "?" + token}


def crop_s1(item, reference):
    arrays, sources = [], {}
    for pol in ("vv", "vh"):
        print(f"    Reading {pol.upper()} COG window", flush=True)
        with open_asset(signed_asset(item["assets"][pol])) as src:
            bounds = transform_bounds(reference.crs, src.crs, *reference.bounds, densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform)
            col, row = int(np.floor(window.col_off)) - 2, int(np.floor(window.row_off)) - 2
            endcol = int(np.ceil(window.col_off + window.width)) + 2
            endrow = int(np.ceil(window.row_off + window.height)) + 2
            if col < 0 or row < 0 or endcol > src.width or endrow > src.height:
                raise ValueError("Crop extends beyond source raster")
            window = Window(col, row, endcol - col, endrow - row)
            source = src.read(1, window=window, masked=True).filled(np.nan)
            target = np.full(reference.shape, np.nan, dtype="float32")
            reproject(source, target, src_transform=src.window_transform(window),
                      src_crs=src.crs, src_nodata=np.nan, dst_transform=reference.transform,
                      dst_crs=reference.crs, dst_nodata=np.nan, resampling=Resampling.nearest)
            sources[pol] = {"crs": src.crs.to_string(), "pixel_spacing": list(src.res),
                            "window": [col, row, window.width, window.height],
                            "transform": list(src.transform)[:6], "nodata": src.nodata,
                            "dtype": src.dtypes[0]}
            arrays.append(target)
    data = np.stack(arrays)
    valid = np.isfinite(data).all(axis=0) & (data > 0).all(axis=0)
    if not valid.all():
        raise ValueError(f"Incomplete positive finite backscatter coverage: {valid.mean():.3%}")
    return data, valid, sources


def acquire_s1(key, meta):
    folder = HERE / "data/sentinel1" / key
    candidates = candidates_for(meta)
    dump(folder / "candidates.json", candidates)
    print(f"{key}: {len(candidates)} dual-polarization candidates, nearest time first", flush=True)
    attempts = []
    for item in candidates:
        print(f"  Checking {item['id']}", flush=True)
        try:
            with rasterio.open(ROOT / "lab1/data/sentinel2" / key / "rgb_reflectance.tif") as ref:
                data, valid, sources = crop_s1(item, ref)
                write_tif(folder / "gamma0_linear.tif", data, ref, ["VV gamma0 linear", "VH gamma0 linear"], np.nan)
                db = (10 * np.log10(data)).astype("float32")
                write_tif(folder / "gamma0_db.tif", db, ref, ["VV gamma0 dB", "VH gamma0 dB"], np.nan)
                write_tif(folder / "valid_mask.tif", valid[None].astype("uint8"), ref, ["1 = positive finite VV and VH"])
        except ValueError as error:
            attempts.append({"scene_id": item["id"], "rejected": str(error)})
            continue
        attempts.append({"scene_id": item["id"], "selected": True})
        dump(folder / "source_item.json", item)
        props = item["properties"]
        record = {"scene_id": item["id"], "source_grd_id": props.get("s1:product_identifier"),
                  "acquired_utc": props["datetime"],
                  "offset_from_sentinel2_hours": (utc(props["datetime"]) - utc(meta["acquired_utc"])).total_seconds() / 3600,
                  "collection": COLLECTION, "product": "Catalyst radiometrically terrain-corrected GRD (RTC)",
                  "polarizations": ["VV", "VH"], "units": "gamma0 linear intensity ratio; dB = 10 log10(linear)",
                  "orbit_state": props["sat:orbit_state"], "relative_orbit": props["sat:relative_orbit"],
                  "instrument_mode": props["sar:instrument_mode"], "sources": sources,
                  "valid_pixels": int(valid.sum()), "excluded_pixels": int(valid.size - valid.sum()),
                  "resampling": "nearest neighbour to exact Sentinel-2 CRS/transform/shape; no speckle filtering",
                  "linear_ranges": {p: [float(data[i].min()), float(data[i].max())] for i, p in enumerate(("VV", "VH"))},
                  "attempts": attempts,
                  "paths": {name: relative(folder / filename) for name, filename in {
                      "linear": "gamma0_linear.tif", "db": "gamma0_db.tif", "mask": "valid_mask.tif",
                      "source_item": "source_item.json", "candidates": "candidates.json"}.items()}}
        dump(folder / "metadata.json", record)
        record["paths"]["metadata"] = relative(folder / "metadata.json")
        return record
    dump(folder / "attempts.json", attempts)
    raise RuntimeError(f"No fully valid S1 scene for {key}")


def load_inputs():
    """Verify provenance hashes, grids, masks and values; return arrays for analysis."""
    manifest = json.loads(MANIFEST.read_text())
    for path, expected in manifest["sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    inputs = {}
    for key, record in manifest["regions"].items():
        values = {}
        reference = None
        for sensor, files in (("sentinel2", ("reflectance", "dn", "mask")),
                              ("sentinel1", ("linear", "db", "mask"))):
            for field in files:
                with rasterio.open(ROOT / record[sensor]["paths"][field]) as src:
                    grid = (src.crs, src.transform, src.shape)
                    reference = reference or grid
                    assert grid == reference, f"Grid mismatch: {key}/{sensor}/{field}"
                    values[f"{sensor}_{field}"] = src.read() if field != "mask" else src.read(1).astype(bool)
        s2mask, s1mask = values["sentinel2_mask"], values["sentinel1_mask"]
        assert s2mask.any() and s1mask.any()
        assert np.isfinite(values["sentinel2_reflectance"][:, s2mask]).all()
        linear = values["sentinel1_linear"][:, s1mask]
        assert np.isfinite(linear).all() and (linear > 0).all()
        np.testing.assert_allclose(values["sentinel1_db"][:, s1mask], 10 * np.log10(linear), atol=1e-5)
        inputs[key] = values
    return manifest, inputs


def prepare(refresh=False):
    if MANIFEST.exists() and not refresh:
        manifest, _ = load_inputs()
        print("Verified saved lab 2 inputs offline", flush=True)
        return manifest
    regions = {}
    for key in AOIS:
        meta, s2 = prepare_s2(key)
        s1 = acquire_s1(key, meta)
        regions[key] = {"name": meta["aoi"]["name"], "terrain": meta["aoi"]["terrain"],
                        "sentinel2": s2, "sentinel1": s1}
    manifest = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "regions": regions,
                "catalogue": CATALOGUE, "selection": "nearest UTC acquisition within +/-12 days, VV+VH, full valid crop",
                "sentinel2_mask": {"excluded_SCL": list(BAD_SCL), "rule": "all bands finite, unmasked, DN>0; retain negative reflectance"},
                "sentinel1_mask": "finite positive gamma0 in both VV and VH; not an independent layover/shadow classification",
                "attribution": "Contains modified Copernicus Sentinel data 2025. S1 processed by Catalyst, hosted by Microsoft Planetary Computer (CC-BY-4.0); S2 accessed through Element 84 Earth Search."}
    paths = {p for r in regions.values() for s in ("sentinel1", "sentinel2") for p in r[s]["paths"].values()}
    manifest["sha256"] = {p: digest(ROOT / p) for p in sorted(paths)}
    dump(MANIFEST, manifest)
    load_inputs()
    print("Prepared and verified both regions", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Repeat catalogue search/download; may change scenes")
    prepare(parser.parse_args().refresh)
