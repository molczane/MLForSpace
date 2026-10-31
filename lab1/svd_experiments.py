"""Refined rank selection and equal-count omitted-component experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

SSIM_MIN = 0.99
PSNR_MIN = 40.0
SHIFT = 10


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(factor, start, count):
    U, s, Vh = factor
    if start < 0 or count < 1 or start + count > len(s):
        raise ValueError("Component interval exceeds the SVD")
    return (U[:, start:start+count] * s[start:start+count]) @ Vh[start:start+count]


def measure(original, reconstructed, s, start, count):
    mse = float(np.mean((original-reconstructed)**2))
    omitted = np.concatenate((s[:start], s[start+count:]))
    np.testing.assert_allclose(mse, np.sum(omitted**2)/original.size, rtol=1e-8, atol=1e-18)
    return {"k": count, "first_component": start+1, "last_component": start+count,
            "retained_energy": float(np.sum(s[start:start+count]**2)/np.sum(s**2)),
            "mse": mse, "psnr_db": float(10*np.log10(255**2/mse)) if mse else float("inf"),
            "ssim": float(structural_similarity(original, reconstructed, data_range=255,
                          win_size=7, gaussian_weights=False, use_sample_covariance=True)),
            "mean_gray": float(reconstructed.mean()),
            "mean_absolute_error": float(np.mean(np.abs(original-reconstructed))),
            "max_absolute_error": float(np.max(np.abs(original-reconstructed))),
            "fraction_negative": float(np.mean(reconstructed < 0))}


def finish_figure(fig, path):
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)


def run_experiments(images, factors, config, output, input_hashes):
    output = Path(output)
    refined, selected = [], {}
    for aoi in config["aois"]:
        key = aoi["id"]
        original = images[key]
        U, s, Vh = factors[key]
        current = np.zeros_like(original)
        last_mse = float("inf")
        for k in range(1, len(s)+1):
            # An orthogonal rank-one update avoids 512 separate matrix products.
            current += s[k-1] * np.outer(U[:, k-1], Vh[k-1])
            row = {"aoi": key, "terrain": aoi["terrain"],
                   **measure(original, current, s, 0, k)}
            assert row["mse"] <= last_mse + 1e-10
            last_mse = row["mse"]
            row["passes_numeric_rule"] = row["ssim"] >= SSIM_MIN and row["psnr_db"] >= PSNR_MIN
            refined.append(row)
            if row["passes_numeric_rule"] and key not in selected:
                selected[key] = row
        np.testing.assert_allclose(current, original, rtol=0, atol=1e-9)
        print(f"{aoi['name']}: first numeric pass k={selected[key]['k']}, "
              f"SSIM={selected[key]['ssim']:.6f}, PSNR={selected[key]['psnr_db']:.2f} dB")
    write_csv(output / "refined_metrics.csv", refined)

    # Full frame, common central detail patch, and absolute error at nearby ranks.
    for aoi in config["aois"]:
        key = aoi["id"]
        k = selected[key]["k"]
        ranks = sorted(set((max(1, k-8), max(1, k-1), k, min(512, k+8))))
        panels = [("Original", images[key])] + [
            (f"k = {rank}" + (" · candidate" if rank == k else ""), reconstruct(factors[key], 0, rank))
            for rank in ranks]
        fig, axes = plt.subplots(3, len(panels), figsize=(20, 11), layout="constrained")
        for i, (label, image) in enumerate(panels):
            axes[0,i].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[0,i].set_title(label)
            axes[1,i].imshow(image[176:336,176:336], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[1,i].set_title("Centre patch [176:336, 176:336]")
            error = np.abs(images[key]-image)
            error_plot = axes[2,i].imshow(error[176:336,176:336], cmap="inferno", vmin=0, vmax=10, interpolation="nearest")
            axes[2,i].set_title("Patch absolute error · 0–10 gray levels")
        for ax in axes.flat:
            ax.axis("off")
        fig.colorbar(error_plot, ax=list(axes[2]), shrink=.6, label="Absolute error (gray levels)", location="bottom")
        fig.suptitle(f"{aoi['name']} — refined visual review\n"
                     "Top: full image; middle: enlarged detail; bottom: amplified error diagnostic. "
                     "Grayscale limits fixed at 0–255.")
        finish_figure(fig, output / f"{key}_rank_review.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for aoi in config["aois"]:
        key = aoi["id"]
        values = [r for r in refined if r["aoi"] == key and r["k"] <= 350]
        for ax, metric in zip(axes, ("ssim", "psnr_db")):
            ax.plot([r["k"] for r in values], [r[metric] for r in values], label=aoi["name"])
            ax.scatter([selected[key]["k"]], [selected[key][metric]], s=30)
    axes[0].axhline(SSIM_MIN, color="black", linestyle="--")
    axes[0].set(ylabel="SSIM", ylim=(.94, 1.002))
    axes[1].axhline(PSNR_MIN, color="black", linestyle="--")
    axes[1].set(ylabel="PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("Rank k (every integer tested; view limited to 1–350)")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Refined numerical screening · dots show first joint pass")
    finish_figure(fig, output / "refined_quality.png")

    comparisons = []
    for aoi in config["aois"]:
        key = aoi["id"]
        s = factors[key][1]
        ranks = sorted(set((1, 5, 10, 20, 40, 80, 120, 160, 256, 502, selected[key]["k"])))
        assert max(ranks)+SHIFT <= len(s)
        for k in ranks:
            pair = []
            for mode, start in (("leading", 0), ("skip10", SHIFT)):
                image = reconstruct(factors[key], start, k)
                row = {"aoi": key, "mode": mode, **measure(images[key], image, s, start, k)}
                comparisons.append(row)
                pair.append(row)
            assert pair[1]["mse"] + 1e-9 >= pair[0]["mse"]
        k = selected[key]["k"]
        leading = reconstruct(factors[key], 0, k)
        skipped = reconstruct(factors[key], SHIFT, k)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), layout="constrained")
        for ax, label, image in zip(axes[:3], ("Original", f"Leading: 1–{k}", f"Skip ten: 11–{k+10}"),
                                    (images[key], leading, skipped)):
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{label}\nmean gray = {image.mean():.2f}")
            ax.axis("off")
        signed = axes[3].imshow(skipped, cmap="RdBu_r", vmin=-30, vmax=30, interpolation="nearest")
        axes[3].set_title("Skip ten: signed diagnostic\nfixed scale −30 to +30; 0 is white")
        axes[3].axis("off")
        fig.colorbar(signed, ax=axes[3], shrink=.75, label="Gray-level contribution")
        fig.suptitle(f"{aoi['name']} — equal component count k={k}; grayscale limits 0–255\n"
                     "The diverging panel only reveals residual signed structure; it is not a restored image.")
        finish_figure(fig, output / f"{key}_skip10.png")
    write_csv(output / "skip10_metrics.csv", comparisons)
    write_csv(output / "selected_ranks.csv", list(selected.values()))
    manifest = json.loads((output / "run_metadata.json").read_text())
    manifest.update({"refined_ranks": "Every integer 1 through 512, per AOI",
                     "acceptance_screen": {"ssim_min": SSIM_MIN, "psnr_min_db": PSNR_MIN},
                     "skip_components": SHIFT,
                     "shifted_metrics": "Raw signed contributions; no clipping or mean restoration",
                     "refined_input_sha256": input_hashes})
    (output / "run_metadata.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return refined, selected, comparisons


def write_submission_report(refined, selected, comparisons, review_path, output, config, input_hashes):
    """Combine reproducible measurements with a scene-specific human-readable review."""
    output = Path(output)
    review = json.loads(Path(review_path).read_text())
    accepted = []
    for aoi in config["aois"]:
        key = aoi["id"]
        note = review[key]
        if note["input_sha256"] != input_hashes[key] or note["accepted_k"] != selected[key]["k"]:
            raise ValueError(f"Inputs or candidate changed for {key}; repeat visual review before accepting")
        assert selected[key]["passes_numeric_rule"]
        assert all(not row["passes_numeric_rule"] for row in refined
                   if row["aoi"] == key and row["k"] < note["accepted_k"])
        accepted.append({**selected[key], "accepted_for_lab": True,
                         "visual_review": note["visual_review"], "remaining_differences": note["remaining_differences"]})
    write_csv(output / "acceptance_results.csv", accepted)
    report = ["# SVD lab: results and observations", "",
              "Executed notebook: [algebra.ipynb](../../algebra.ipynb). "
              "Reproduction instructions: [lab README](../../README.md).", "",
              "## Accepted ranks for the selected images", "",
              "Acceptance combines SSIM ≥ 0.99, PSNR ≥ 40 dB and informal visual inspection "
              "of the full frame and enlarged central patch at fixed grayscale limits. "
              "This is a conservative rule for this lab, not proof of perceptual identity. "
              "Every integer rank 1–512 was evaluated; these are the smallest tested ranks "
              "passing the joint rule and accepted in the recorded review.", "",
              "| Terrain / AOI | Accepted k | Energy (%) | PSNR (dB) | SSIM | Previous k SSIM |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for aoi in config["aois"]:
        key = aoi["id"]
        row = selected[key]
        previous = next(r for r in refined if r["aoi"] == key and r["k"] == row["k"]-1)
        report.append(f"| {aoi['name']} ({aoi['terrain']}) | {row['k']} | "
                      f"{100*row['retained_energy']:.5f} | {row['psnr_db']:.3f} | {row['ssim']:.6f} | {previous['ssim']:.6f} |")
    report += ["", "The previous rank can look almost identical while narrowly missing the numerical threshold. "
               "These values are operational choices, not measured human just-noticeable differences.", ""]
    for aoi in config["aois"]:
        note = review[aoi["id"]]
        report += [f"### {aoi['name']}", "", note["visual_review"], "",
                   "Remaining differences: " + note["remaining_differences"], "",
                   f"![{aoi['name']}: nearby ranks and detail crops]({aoi['id']}_rank_review.png)", ""]
    report += ["## Does skipping the first ten components help?", "",
               "No improvement was found in the tested equal-count comparisons. "
               "Components 1…k are compared with 11…k+10; both retain exactly k components. "
               "The maximum shifted count is 502, because the input has 512 components. "
               "The following rows use each terrain's accepted k.", "",
               "| AOI | k | Leading PSNR | Skip-ten PSNR | Leading SSIM | Skip-ten SSIM | Leading mean gray | Skip-ten mean gray |",
               "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for aoi in config["aois"]:
        key = aoi["id"]
        k = selected[key]["k"]
        leading = next(r for r in comparisons if r["aoi"] == key and r["k"] == k and r["mode"] == "leading")
        skip = next(r for r in comparisons if r["aoi"] == key and r["k"] == k and r["mode"] == "skip10")
        report.append(f"| {aoi['name']} | {k} | {leading['psnr_db']:.3f} | {skip['psnr_db']:.3f} | "
                      f"{leading['ssim']:.6f} | {skip['ssim']:.6f} | {leading['mean_gray']:.3f} | {skip['mean_gray']:.5f} |")
    report += ["", "The leading components carry mean brightness and major spatial structures. "
               "Removing them leaves small signed contributions near zero. Their fixed-range grayscale "
               "display is dark; the diverging diagnostic exposes residual structure but does not restore "
               "the missing brightness or structures. Metrics are computed before clipping or rounding.", "",
               "The leading-k SVD reconstruction minimises squared Frobenius error among rank-k matrices. "
               "Skipping large singular values therefore cannot lower MSE or improve PSNR with a fixed "
               "data range. SSIM has no such ordering theorem, so its outcome is reported empirically. "
               "Even retaining all 502 shifted components cannot recover the omitted leading ten.", ""]
    for aoi in config["aois"]:
        report += [f"![{aoi['name']}: equal-count omitted-component comparison]({aoi['id']}_skip10.png)", ""]
    report += ["## Energy, quality and storage", "",
               "At rank 1, Kraków already retains about 96% of uncentered energy while SSIM is only "
               "0.202. Mean brightness dominates energy, especially in the sea image. Energy alone "
               "does not establish local-detail preservation. Low sea contrast also makes absolute pixel "
               "error small; neither high PSNR nor SSIM should replace image inspection.", "",
               "![Refined quality curves](refined_quality.png)", "",
               "A rank-k factorisation stores k(m+n+1) scalar values. For a 512-square image the scalar "
               "break-even point is k < 262144/1025 ≈ 255.75. High fidelity can require enough components "
               "to lose even this scalar-count saving. This is not a PNG byte-size comparison: float64 "
               "factors use more bytes per scalar than the original uint8 pixels, and real codecs encode "
               "them differently.", "",
               "## Scope and reproducibility", "",
               "Inputs: Kraków 2025-08-13, Białowieża 2025-07-02, Baltic 2025-08-19; all 512 × 512 "
               "at native 10 m spacing. These are three selected crops with different dates, illumination, "
               "atmosphere and noise. Results do not establish universal terrain requirements or a formal "
               "human perceptual threshold. The experiment uses display grayscale PNGs, not raw reflectance.", "",
               "Metrics: float64, no centering, rounding or clipping; PSNR data_range=255; SSIM "
               "data_range=255, win_size=7, gaussian_weights=False, use_sample_covariance=True. "
               "Image displays always use vmin=0, vmax=255, nearest interpolation. Error and signed "
               "diagnostics are explicitly labelled and have common fixed scales.", "",
               "Checks: full-rank recovery within 1e-9 gray levels; nonincreasing leading-rank MSE; "
               "agreement of measured MSE with omitted squared singular values for every reconstruction; "
               "equal component counts; and shifted MSE never below leading MSE. Input hashes and "
               "visual-review hashes are verified before final acceptance.", "",
               "## Files", "",
               "- [Acceptance table and review notes](acceptance_results.csv)",
               "- [All 1,536 refined measurements](refined_metrics.csv)",
               "- [All paired leading / skip-ten measurements](skip10_metrics.csv)",
               "- [Initial coarse measurements](metrics.csv)",
               "- [All singular values](singular_values.csv)",
               "- [Run settings, versions and input hashes](run_metadata.json)", "",
               "![Grayscale originals](originals.png)",
               "![Singular-value decay](singular_values.png)", ""]
    for aoi in config["aois"]:
        report += [f"![{aoi['name']}: original coarse rank grid]({aoi['id']}_reconstructions.png)", ""]
    report += ["Contains modified Copernicus Sentinel data 2025, accessed through Element 84 Earth Search.", ""]
    (output / "README.md").write_text("\n".join(report))
    manifest = json.loads((output / "run_metadata.json").read_text())
    manifest["visual_review"] = review
    (output / "run_metadata.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    return accepted
