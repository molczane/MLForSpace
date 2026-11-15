# Lab 3 submission review

The main deliverable is [analysis.ipynb](analysis.ipynb), with saved figures,
parameters, class maps, comparisons and written answers. The notebook follows
the six phases in [CHECKLIST.md](CHECKLIST.md). All classifiers are implemented
with NumPy operations in [bayes.py](bayes.py) and [knn.py](knn.py);
[evaluation.py](evaluation.py) computes metrics from explicit confusion counts.

## Assignment coverage

| Assignment requirement | Evidence in the notebook |
| --- | --- |
| Forest/urban images and pure training areas | Section 1: original Kraków/Białowieża previews, manual ROI overlays and enlarged chips; exact labels/samples saved |
| New image | Section 1: separate Tychy scene/footprint, acquisition and QA; no training samples from this image |
| Histograms and parametric PDFs/parameters | Section 2: six class/band distributions, fitted Gaussian marginals, raw/scaled means and full covariances, priors and fit assessment |
| Bayes classifier from scratch | Section 3: posterior decision rule, Cholesky log scores, exact ties, all-valid-pixel prediction and red/blue maps |
| Explain Bayes results | Section 3: reference errors, feature shift, shadows/mixed pixels, scene differences and limitations |
| kNN classifier from scratch | Section 4: explicit distances, stable neighbour order, majority voting, bounded batches and maps for k = 1/3/5 |
| Discuss changing k | Sections 4–5: change maps/counts, paired reference corrections/errors, vote-reversal example and nonmonotonic effects |
| Present/compare results | Section 5: common-reference confusion matrices, precision/recall/F1, accuracy, balanced accuracy, assumptions and cost |
| Reproducible submission | Section 6: saved-artifact audit, relocated offline copy, source/input hashes and archive instructions |

## Scientific outcome and limits

All four models use the same 1,024 saved training samples, three reflectance
features and training-only scaling. The evaluation uses the same 2,304 forest
and 256 urban reference pixels. Balanced accuracy is **53.212% for Bayes**,
**99.674% for kNN k = 1/3** and **99.653% for k = 5**.

Bayes recalls only 148/2,304 forest references. This poor transfer is retained
and explained, rather than being hidden or retuned away. k = 1 → 3 changes no
reference decisions; k = 3 → 5 adds one forest-reference error. Increasing k is
not a spatial smoothing operation and does not improve accuracy monotonically.

The manual references cover about 0.98% of valid test pixels and emphasize
clear canopy/industrial interiors. They are spatially correlated and not
field verified. Training class is confounded with location/date; a single new
scene does not establish general performance. Fields, gardens, water and mixed
pixels receive forced binary predictions without full-image ground truth.
Numerical correctness is separate from scientific accuracy.

## Final verification

The final notebook has **30 code cells**, including the final audit cell.
A fresh-kernel sequential run, saved outputs, source/written-answer review and
figure inspection are recorded in
[results/notebook_review.json](results/notebook_review.json).
The artifact and relocated-copy outcomes are recorded in
[results/submission_checks.json](results/submission_checks.json).

[submission.py](submission.py) independently checks the recorded input/output
hashes, source calibration/grids/QA, exact seeded sample identities, split
buffer and scaling; nine histogram exports and fitted Gaussian moments; all
four prediction rasters, colour/nodata encoding and georeferencing; all Bayes
predictions/scores; saved kNN distances/votes and 128 fresh exhaustive queries;
common-reference reports, paired k changes, 15 scientific figures and local
documentation links. Earlier notebook cells also check hand-calculated
algorithms, exact ties, integer overflow and full-image batch invariance.

The relocated audit copies the submission payload to a temporary directory,
loads helper modules from that directory and repeats the artifact checks with
network connections disabled. It tests saved-artifact/source portability; the
complete notebook execution is verified separately through the IDE. No data
download or private credentials are needed with the saved inputs.

## Portable ZIP

From the repository root, after saving final notebook outputs:

```sh
uv run python lab3/submission.py --package
```

The output is `dist/MLForSpace-lab3-submission.zip`, with a
`MLForSpace-lab3/` top-level directory. It contains the notebook, code,
labels/samples/scaling, input rasters, results, documentation and dependency
files. All three labs are included to retain reused inputs and cross-lab
navigation/provenance links. Environments, Git/editor metadata, caches and
`dist/` are excluded. The root `.gitignore` is included deliberately.

The builder verifies ZIP CRCs, every included byte sequence against its source,
the extracted `SUBMISSION_FILES.sha256` inventory and extracted documentation
links. Companion `dist/MLForSpace-lab3-submission.zip.sha256` and
`dist/MLForSpace-lab3-archive-checks.json` record archive integrity. The archive
and companions are local deliverables ignored by Git; the build code, scientific
inputs/results and review records are staged for version control.

Extract the ZIP, enter `MLForSpace-lab3/`, run `uv sync --locked`, select the
resulting `.venv` kernel and open `lab3/analysis.ipynb`. Notebook execution uses
saved local data. Dependency installation needs a package cache or network.
Saved notebook outputs and standalone figures can be read without execution.
