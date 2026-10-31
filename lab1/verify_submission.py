"""Verify the executed notebook, input integrity, result tables and report links."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import nbformat
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parent
OUTPUT = LAB / "results/svd"


def read_csv(name):
    with (OUTPUT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    notebook = nbformat.read(LAB / "algebra.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code)+1))
    assert all(out.output_type != "error" for cell in code for out in cell.outputs)
    embedded = sum("image/png" in out.get("data", {}) for cell in code for out in cell.outputs)
    assert embedded == 13, f"Expected 13 embedded figures, found {embedded}"

    config = json.loads((LAB / "sentinel2_aois.json").read_text())
    review = json.loads((LAB / "visual_review.json").read_text())
    manifest = json.loads((OUTPUT / "run_metadata.json").read_text())
    refined = read_csv("refined_metrics.csv")
    accepted = read_csv("acceptance_results.csv")
    comparisons = read_csv("skip10_metrics.csv")
    assert len(refined) == 1536 and len(accepted) == 3 and len(comparisons) == 66
    assert len(read_csv("metrics.csv")) == 30
    for aoi in config["aois"]:
        key = aoi["id"]
        source = LAB / "data/sentinel2" / key
        metadata = json.loads((source / "metadata.json").read_text())
        for name, expected in metadata["sha256"].items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
        digest = metadata["sha256"]["grayscale.png"]
        assert digest == manifest["input_sha256"][key] == review[key]["input_sha256"]
        series = [row for row in refined if row["aoi"] == key]
        assert [int(row["k"]) for row in series] == list(range(1, 513))
        assert np.all(np.diff([float(row["mse"]) for row in series]) <= 1e-10)
        first = next(row for row in series if float(row["ssim"]) >= .99 and float(row["psnr_db"]) >= 40)
        choice = next(row for row in accepted if row["aoi"] == key)
        assert int(first["k"]) == int(choice["k"]) == review[key]["accepted_k"]
        assert choice["accepted_for_lab"] == "True"
        assert choice["remaining_differences"] == review[key]["remaining_differences"]
        for name in ("k", "mse", "ssim", "psnr_db", "retained_energy"):
            assert choice[name] == first[name]
        pairs = [row for row in comparisons if row["aoi"] == key]
        for k in {int(row["k"]) for row in pairs}:
            leading = next(r for r in pairs if int(r["k"]) == k and r["mode"] == "leading")
            skipped = next(r for r in pairs if int(r["k"]) == k and r["mode"] == "skip10")
            assert int(leading["first_component"]) == 1 and int(leading["last_component"]) == k
            assert int(skipped["first_component"]) == 11 and int(skipped["last_component"]) == k+10 <= 512
            assert float(skipped["mse"]) >= float(leading["mse"])-1e-9
            assert float(skipped["psnr_db"]) <= float(leading["psnr_db"])
            assert float(skipped["ssim"]) < float(leading["ssim"])
    for path in OUTPUT.glob("*.png"):
        with Image.open(path) as image:
            image.verify()
    for path in (LAB.parent / "README.md", LAB / "README.md", OUTPUT / "README.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in link and not link.startswith("#"):
                assert (path.parent / link.split("#")[0]).exists(), f"Broken link in {path}: {link}"
    print(f"PASS: {len(code)} executed code cells, 13 embedded figures, 1,536 dense rank measurements, "
          "66 paired measurements, 3 accepted ranks, input/review hashes and report links.")


if __name__ == "__main__":
    main()
