"""Download matching native-resolution Sentinel-2 crops from public COGs.

Run from the repository root: uv run python lab1/prepare_sentinel2.py
No authentication, SVD, mosaicking, RGB resizing, or per-image stretching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import rasterio
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, from_bounds

from http_range import open_asset

HERE = Path(__file__).resolve().parent
BANDS = ("red", "green", "blue")
BAD_SCL = (0, 1, 3, 7, 8, 9, 10, 11)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def query(config, aoi):
    # Query the centre; full rectangular coverage is checked against each raster.
    lon, lat = aoi["longitude"], aoi["latitude"]
    body = {"collections": [config["collection"]], "bbox": [lon-0.001, lat-0.001, lon+0.001, lat+0.001],
            "datetime": config["datetime"], "limit": 100,
            "query": {"eo:cloud_cover": {"lte": config["max_scene_cloud_percent"]}}}
    url, features = config["catalogue"] + "/search", []
    while url:
        request = Request(url, data=json.dumps(body).encode() if body else None,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            page = json.load(response)
        features.extend(page["features"])
        link = next((x for x in page.get("links", []) if x["rel"] == "next"), None)
        url = link["href"] if link else None
        if link:
            body = ({**body, **link.get("body", {})} if link.get("merge") else link.get("body")) if link.get("method", "GET") == "POST" else None
    return sorted(features, key=lambda f: (f["properties"]["eo:cloud_cover"], f["id"]))


def grid(item, aoi, config):
    with open_asset(item["assets"]["red"]) as source:
        if source.res != (config["resolution_m"],) * 2:
            raise ValueError("Unexpected native pixel resolution")
        x, y = Transformer.from_crs(4326, source.crs, always_xy=True).transform(aoi["longitude"], aoi["latitude"])
        row, col = source.index(x, y)
        size = config["size_pixels"]
        window = Window(col-size//2, row-size//2, size, size)
        if window.col_off < 0 or window.row_off < 0 or window.col_off+size > source.width or window.row_off+size > source.height:
            raise ValueError("AOI crosses tile edge")
        return window, source.window_transform(window), source.crs, window_bounds(window, source.transform)


def read_scl(item, bounds, crs, size):
    with open_asset(item["assets"]["scl"]) as source:
        if source.crs != crs:
            raise ValueError("SCL CRS differs from RGB")
        # Only categorical QA is sampled from 20 m to 10 m, using nearest neighbour.
        return source.read(1, window=from_bounds(*bounds, transform=source.transform),
                           out_shape=(size, size), resampling=Resampling.nearest)


def read_rgb(item, window, transform, crs):
    raw, reflectance, conversions = [], [], {}
    for band in BANDS:
        asset = item["assets"][band]
        with open_asset(asset) as source:
            if source.crs != crs or source.window_transform(window) != transform:
                raise ValueError("RGB grids do not align")
            data = source.read(1, window=window, masked=True)
            if np.ma.getmaskarray(data).any():
                raise ValueError("RGB contains nodata")
            data = np.asarray(data)
        spec = asset["raster:bands"][0]
        scale = spec["scale"]
        # Collection 1 has consistent scale/offset metadata; reject legacy items.
        if item["collection"] != "sentinel-2-c1-l2a":
            raise ValueError("Use Collection 1 to avoid ambiguous legacy BOA offsets")
        offset = spec.get("offset", 0.0)
        raw.append(data)
        reflectance.append(data.astype("float32") * scale + offset)
        conversions[band] = {"scale": scale, "effective_offset": offset,
                             "catalogue_offset": spec.get("offset", 0.0)}
    return np.stack(raw), np.stack(reflectance), conversions


def write_tif(path, data, crs, transform, descriptions):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=data.shape[0], dtype=data.dtype, crs=crs, transform=transform,
                       compress="deflate") as output:
        output.write(data)
        for i, name in enumerate(descriptions, 1):
            output.set_band_description(i, name)


def acquire(config, aoi, output):
    folder = output / aoi["id"]
    candidates = query(config, aoi)
    dump(folder / "candidates.json", candidates)
    print(f"{aoi['name']}: {len(candidates)} candidate scenes", flush=True)
    attempts = []
    for item in candidates:
        print(f"  Checking {item['id']}", flush=True)
        try:
            window, transform, crs, bounds = grid(item, aoi, config)
            scl = read_scl(item, bounds, crs, config["size_pixels"])
            bad = float(np.isin(scl, BAD_SCL).mean())
            attempts.append({"scene": item["id"], "bad_fraction": bad})
            if bad > config["max_aoi_bad_fraction"]:
                continue
            print(f"  Local QA passed ({bad:.3%} bad); reading RGB", flush=True)
            raw, reflectance, conversions = read_rgb(item, window, transform, crs)
        except ValueError as error:
            attempts.append({"scene": item["id"], "rejected": str(error)})
            continue
        # Fixed conversion shared by all three images. No percentile stretching.
        rgb = np.moveaxis(np.rint(255 * np.clip(reflectance / config["display_reflectance_max"], 0, 1)
                                 ** (1 / config["display_gamma"])).astype("uint8"), 0, -1)
        write_tif(folder / "rgb_dn.tif", raw, crs, transform, BANDS)
        write_tif(folder / "rgb_reflectance.tif", reflectance, crs, transform, BANDS)
        write_tif(folder / "scl.tif", scl[None], crs, transform, ["SCL nearest sampled from 20m"])
        Image.fromarray(rgb).save(folder / "rgb.png")
        Image.fromarray(rgb).convert("L").save(folder / "grayscale.png")
        labels, counts = np.unique(scl, return_counts=True)
        fractions = {str(int(k)): float(v/scl.size) for k, v in zip(labels, counts)}
        west, south, east, north = bounds
        corners = [(west, south), (east, south), (east, north), (west, north), (west, south)]
        projection = Transformer.from_crs(crs, 4326, always_xy=True)
        polygon = {"type": "Polygon", "coordinates": [[list(projection.transform(x, y)) for x, y in corners]]}
        metadata = {"aoi": aoi, "scene_id": item["id"], "acquired_utc": item["properties"]["datetime"],
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
                    "crs": crs.to_string(), "bounds": bounds, "transform": list(transform)[:6],
                    "window": {"col": window.col_off, "row": window.row_off, "size": window.width},
                    "geometry": polygon, "aoi_bad_fraction": bad, "scl_fractions": fractions,
                    "conversions": conversions, "attempts": attempts,
                    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix in (".tif", ".png")}}
        dump(folder / "source_item.json", item)
        dump(folder / "metadata.json", metadata)
        print(f"  Selected {item['id']}; bad pixels {bad:.3%}", flush=True)
        return metadata
    dump(folder / "attempts.json", attempts)
    raise RuntimeError(f"No acceptable fully covered scene for {aoi['id']}; see {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "sentinel2_aois.json")
    parser.add_argument("--output", type=Path, default=HERE / "data/sentinel2")
    parser.add_argument("--refresh", action="store_true", help="Repeat search and download; default reuses saved matching results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["size_pixels"] <= 0 or config["size_pixels"] % 2:
        raise ValueError("size_pixels must be a positive even integer")
    results = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                      GDAL_HTTP_TIMEOUT="180", GDAL_HTTP_MAX_RETRY="3",
                      GDAL_HTTP_MULTIRANGE="SERIAL", GDAL_HTTP_VERSION="1.1"):
        for aoi in config["aois"]:
            saved = args.output / aoi["id"] / "metadata.json"
            if saved.exists() and not args.refresh:
                metadata = json.loads(saved.read_text())
                if metadata["configuration"] != config:
                    raise ValueError("Configuration changed; use --refresh or another --output directory")
                for name, expected in metadata["sha256"].items():
                    if hashlib.sha256((saved.parent/name).read_bytes()).hexdigest() != expected:
                        raise ValueError(f"Changed output: {saved.parent/name}")
                print(f"{aoi['name']}: verified saved outputs", flush=True)
            else:
                metadata = acquire(config, aoi, args.output)
            results.append(metadata)
    dump(args.output / "aois.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": m["geometry"], "properties": {**m["aoi"], "scene_id": m["scene_id"]}} for m in results]})
    size = config["size_pixels"]
    preview = Image.new("RGB", (3*size, size+55), "white")
    draw = ImageDraw.Draw(preview)
    font_path = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    title_font = ImageFont.truetype(font_path, 16)
    detail_font = ImageFont.truetype(font_path, 13)
    for i, m in enumerate(results):
        with Image.open(args.output/m["aoi"]["id"]/"rgb.png") as image:
            preview.paste(image, (i*size, 55))
        draw.text((i*size+10, 5), f"{m['aoi']['name']} ({m['aoi']['terrain']})", fill="black", font=title_font)
        draw.text((i*size+10, 29), f"{m['acquired_utc'][:10]} | {size} x {size} | 10 m/px", fill="black", font=detail_font)
    preview.save(args.output / "comparison.png")


if __name__ == "__main__":
    main()
