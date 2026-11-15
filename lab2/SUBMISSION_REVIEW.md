# Lab 2 submission review

The main deliverable is [analysis.ipynb](analysis.ipynb), including saved figures,
tables, written answers and the optional compressive-sensing exercise. The
original [Basic_opt.ipynb](Basic_opt.ipynb) is an unchanged reference; its historical
outputs and installation cell are not the reproduction workflow for this lab.

## Assignment coverage

| Assignment requirement | Evidence in the main notebook |
| --- | --- |
| Three Sentinel-2 bands in urban and natural regions | Sections 1–2: B04/B03/B02 for Kraków and Białowieża, six optical distributions, labelled previews |
| Repeat for Sentinel-1 in the same regions | Sections 1–2: VV/VH RTC crops, linear and dB histograms, observations and sensor comparison |
| Record histogram observations | Section 2: peaks, spread, tails, skewness, numeric summaries and acquisition/mask caveats |
| Randomly replace each pixel's two LSBs | Section 3: original uint16 B04, independent seeded draws for both regions, bit and error checks |
| Assess visual quality | Section 3: matched full images, enlarged patches, signed differences and measured errors |
| Compare PDFs/histograms and discuss detection | Section 3: coarse and single-DN histograms, normalization, low-bit diagnostics, explicit answers with limitations |
| Optional compressive sensing and usefulness in EO | Section 4: all supplied formulations, convergence diagnostics, corrected comparison, sparse control and sourced EO discussion |

## Verified outcome

- **17/17 code cells:** fresh kernel, sequential execution, no notebook errors;
  outputs saved. The original tutorial is unchanged.
- **26 input hashes**, **14 histogram exports**, **2 LSB regions**, **5 CS
  reconstruction/baseline arrays**, and **12 decoded scientific figures** passed.
- Source/result checks passed again on an isolated relocated copy with network
  calls disabled. Local links resolve and the locked environment sync succeeds.
- Figure layouts and written answers were reviewed. All required questions and
  the optional CS question have explicit evidence-based answers in the notebook.

The execution/review record is [results/notebook_review.json](results/notebook_review.json).
The machine-readable artifact outcome is `results/submission_checks.json`.
The supplied CS solvers' failures remain explicitly reported rather than presented
as successful signal recovery. No quantitative perceptual or general manipulation-
detection guarantee is claimed.

## Verification scope

The notebook's final section calls [submission.py](submission.py). It checks
input hashes and grids, verifies exported histogram counts/densities against the
source rasters, compares saved LSB rasters and error metrics, recomputes saved
CS reconstruction errors, decodes all twelve scientific figures and checks
local documentation links. It also repeats the artifact checks on an isolated
copy with network access disabled. Results are saved in
`results/submission_checks.json` after the checks finish.

Notebook source, written answers, figures and saved cell outputs are reviewed
through the IDE. All cells must run in order after a kernel restart; optimizer
nonconvergence in the supplied CS examples is an explicitly reported result,
not a failed notebook cell. The reproducible workflow is [documented here](README.md).

## Portable submission

After running the notebook, build the ZIP from the repository root:

```sh
uv run python lab2/submission.py --package
```

It produces `dist/MLForSpace-lab2-submission.zip` and verifies its CRCs and
file hashes. The archive contains the main notebook and all local inputs,
results, code, documentation and dependency files, plus the supporting lab 1
files. Lab 1 is included in full to preserve its provenance links; lab 2 analyses
only Kraków and Białowieża. Rebuilds also include `lab3/` when available to keep
the repository README's navigation links valid. `SUBMISSION_FILES.sha256` inside the archive lists
the exact included files and hashes. Environments, Git/editor metadata, caches
and the ZIP itself are excluded.

Extract the ZIP, enter `MLForSpace-lab2/`, run `uv sync --locked`, and open
`lab2/analysis.ipynb` with the resulting `.venv` kernel. Installing packages needs
an available package cache or network; running the analysis with saved inputs
needs neither remote imagery access nor private credentials. Reading the saved
notebook outputs and standalone figures requires no data download.

The ZIP is a local deliverable, not a publication or Git commit. Regenerate it
after changes to keep its contents and checksum inventory current.
