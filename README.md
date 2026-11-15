# ML for Space

Three completed labs on satellite-image analysis. Start with the notebook for
the lab you need; each notebook includes saved outputs and written conclusions.
Reading results needs no Python environment, data download or credentials.

## Choose a lab

| Lab | Topic | Main notebook | Guide and results | Review |
| --- | --- | --- | --- | --- |
| 1 | SVD image reconstruction | [algebra.ipynb](lab1/algebra.ipynb) | [Input/reproduction guide](lab1/README.md) · [SVD report](lab1/results/svd/README.md) | [Acceptance measurements](lab1/results/svd/acceptance_results.csv) |
| 2 | Sentinel-1/2 histograms, two-LSB replacement, compressive sensing | [analysis.ipynb](lab2/analysis.ipynb) | [Guide](lab2/README.md) · [Results](lab2/results/) | [Checklist](lab2/CHECKLIST.md) · [Submission review](lab2/SUBMISSION_REVIEW.md) |
| 3 | Bayes and kNN forest–urban classification from scratch | [analysis.ipynb](lab3/analysis.ipynb) | [Guide](lab3/README.md) · [Comparison results](lab3/results/comparison/) | [Checklist](lab3/CHECKLIST.md) · [Submission review](lab3/SUBMISSION_REVIEW.md) |

## Results at a glance

- **Lab 1:** reconstructs Kraków, Białowieża and Baltic Sentinel-2 crops with SVD.
  The selected ranks are 258, 251 and 175 under the lab's explicit numerical
  and visual acceptance rule. See the [full report](lab1/results/svd/README.md)
  for the leading/skip-ten comparison, quality curves and limitations.
- **Lab 2:** compares optical and SAR distributions for Kraków/Białowieża,
  replaces two low bits reproducibly and runs the optional compressive-sensing
  example. Supplied solver failures and the corrected comparison are reported
  explicitly. See the [submission review](lab2/SUBMISSION_REVIEW.md).
- **Lab 3:** fits full Gaussian class PDFs and implements Bayes and exhaustive
  kNN for k = 1, 3 and 5. On 2,560 sparse manual Tychy reference pixels,
  balanced accuracy is 53.212% for Bayes, 99.674% for k = 1/3 and 99.653% for
  k = 5. The [notebook](lab3/analysis.ipynb) explains poor Bayes forest transfer,
  small nonmonotonic k effects and why these figures are not full-image or
  general regional accuracy estimates.

## Find data, code and figures

| Location | Contents |
| --- | --- |
| [lab1/data/sentinel2/](lab1/data/sentinel2/) | Original small optical crops, display previews and provenance for Kraków, Białowieża and Baltic |
| [lab1/results/svd/](lab1/results/svd/) | SVD reconstructions, quality plots, measurements and report |
| [lab2/data/](lab2/data/) | Optical QA masks, matching Sentinel-1 crops and source manifest |
| [lab2/results/](lab2/results/) | Histograms, LSB comparisons, compressive-sensing results and audit records |
| [lab3/data/](lab3/data/) | Manual labels, exact training samples/scaling, independent Tychy crop and provenance |
| [lab3/results/bayes/](lab3/results/bayes/) | Fitted Gaussian parameters/PDF plots; Bayes maps in `classification/` |
| [lab3/results/knn/](lab3/results/knn/) | k = 1/3/5 maps, nearest-neighbour records and k-change evidence |
| [lab3/results/comparison/](lab3/results/comparison/) | Common-reference metrics, confusion matrices and all-model comparison |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Shared Python requirements and locked dependencies |

Lab 2 reuses Lab 1 optical data. Lab 3 reuses those crops and Lab 2 optical QA
masks, then adds the independent Tychy scene. Keep the repository directory
structure intact. `lab2/Basic_opt.ipynb` is the unchanged source tutorial; use
`lab2/analysis.ipynb` for the completed Lab 2 submission.

## Run the notebooks

From the repository root, install the locked environment:

```sh
uv sync --locked
```

Python 3.12 or later is required by the project. Select the resulting `.venv`
interpreter in your notebook editor, restart the kernel and run the desired
notebook from top to bottom. Lab 2 and Lab 3 support working directories at the
repository root or their own lab directory. Lab 1 also has a runner:

```sh
uv run python lab1/run_notebook.py
```

Saved crops permit offline analysis without private credentials. Installing
uncached packages or intentionally repeating imagery acquisition needs network
access; exact acquisition options and source details are in each lab's guide.

## Verify saved artifacts

After notebook execution, the following commands audit saved files:

```sh
uv run python lab1/verify_submission.py
uv run python lab2/submission.py
uv run python lab3/submission.py
```

Lab 3's final notebook section also checks a relocated portable copy with
network access disabled. Read [its submission review](lab3/SUBMISSION_REVIEW.md)
for verification scope and the remaining scientific limitations.

## Build portable submissions

Run from the repository root after saving notebook outputs:

```sh
uv run python lab2/submission.py --package
uv run python lab3/submission.py --package
```

Outputs are `dist/MLForSpace-lab2-submission.zip` and
`dist/MLForSpace-lab3-submission.zip`. The Lab 3 archive includes all three labs
so earlier-lab data dependencies and documentation links remain available.
Archives include per-file SHA-256 inventories and exclude environments, caches,
editor/Git metadata and the `dist/` directory. ZIPs are local deliverables,
ignored by Git; rebuild after changes. Extraction/run instructions are in the
corresponding submission reviews.

Contains modified Copernicus Sentinel data 2025. Acquisition providers,
processing details, timestamps and attribution are preserved in the
[Lab 1](lab1/README.md), [Lab 2](lab2/README.md) and [Lab 3](lab3/README.md) guides.
