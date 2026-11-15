# Lab 3 — Bayes and kNN completion checklist

Based on the supplied assignment, the previous Lab 2 conversation and the
current repository. Phases 1–2 are complete; checked items identify verified data and PDF preparation.
Lab 3 classifiers and classification results have not yet been implemented.

## Current starting point and decisions

- [x] Review the previous Lab 2 conversation: Kraków and Białowieża were the
  selected urban–natural pair, and phases 1–5 were completed in sequence.
- [x] Locate reusable Sentinel-2 inputs at
  `../lab1/data/sentinel2/{krakow,bialowieza}/`: original DN, BOA reflectance,
  RGB previews, SCL, source metadata and acquisition information.
- [x] Locate existing optical validity masks under
  `../lab2/data/sentinel2/{krakow,bialowieza}/` and histogram examples in
  [the Lab 2 notebook](../lab2/analysis.ipynb) and
  [histogram results](../lab2/results/histograms/).
- [x] Review previous preparation and verification conventions in
  [Lab 1](../lab1/README.md), [Lab 2](../lab2/README.md) and
  [the Lab 2 submission review](../lab2/SUBMISSION_REVIEW.md).
- [x] Create `lab3/` and this checklist.

Planned baseline: reuse Sentinel-2 B04/B03/B02 **BOA reflectance** as each
pixel's three features, fit one multivariate Gaussian per class for Bayes,
and implement Euclidean-distance kNN for k = 1, 3 and 5. Use the same training
samples, feature transformation and test pixels for both methods. NumPy may
provide array operations and linear algebra; implement parameter estimation,
Bayes scores, neighbour selection and voting explicitly. Do not delegate these
steps to ready-made classifiers or neighbour-search estimators.

Sentinel-1, additional bands and more complex PDF families are optional extensions,
not assignment requirements. Preserve earlier labs and their original inputs.

## 1. Prepare the notebook, labelled data and test image — required

Completed in [analysis.ipynb](analysis.ipynb). Kraków/Białowieża supply 512
training samples per class, with separate spatial holdouts and a checked 160 m
buffer. The independent 2025-08-14 Tychy scene provides a new mixed forest–urban
image and 2,560 manually labelled test-reference pixels. All 23 ROIs were
visually inspected; labels are approximate assistant annotations, not field
truth. Source hashes, calibration, masks, split/sample identities and
training-only scaling pass [data/verification.json](data/verification.json).
All seven code cells passed a fresh-kernel run; the saved-input audit passed
with network disabled. See [data/phase1_review.json](data/phase1_review.json).
See [README.md](README.md), [rois.json](rois.json),
[data/manifest.json](data/manifest.json) and the
[labelled source overview](data/labelled_overview.png). The test acquisition's
0.25% local flagged-SCL selection threshold is documented; all 496 flagged
pixels are excluded.

- [x] Create `lab3/analysis.ipynb`, with sections matching phases 1–6 below, and
  `lab3/README.md` with input provenance and reproduction instructions. Reuse
  the project environment in `../pyproject.toml` and `../uv.lock`.
- [x] Verify reusable inputs against their recorded hashes and metadata;
  confirm B04/B03/B02 order, units, dimensions, grid and masks. Keep numerical
  reflectance for classification and the existing fixed display mapping for
  previews; do not train on independently contrast-stretched images.
- [x] Inspect the RGB crops and hand-select several class-pure forest and
  built-up urban ROIs. Save reproducible polygons or pixel coordinates and
  raster label masks, with labelled overlays on the source images. Exclude
  water, fields, parks within urban areas, uncertain boundaries and other
  out-of-scope land covers from the labelled samples.
- [x] Define the lab's meaning of “urban” and the label encoding, for example
  forest = 0, urban = 1 and ignored/unlabelled = 255. Document who/what supplied
  the reference labels. Do not equate an entire Kraków image with urban or an
  entire Białowieża image with forest. SCL vegetation is not a forest label.
- [x] Apply optical QA/nodata masks before extracting samples. Keep validity
  masks separate from land-cover labels; retain valid dark forest pixels.
- [x] Reserve disjoint spatial ROIs/blocks for training and evaluation before
  fitting anything. Avoid a random pixel split that puts neighbouring pixels
  from the same training patch into the test set. Record any spatial buffer,
  sampling seed and per-class sample counts.
- [x] Obtain a **new mixed forest–urban test image** that contributes no
  training samples, preferably a separate Sentinel-2 scene/area with comparable
  processing and season. Record its source, acquisition date, bands, units,
  footprint, quality mask and preprocessing. A collage of training patches
  may illustrate the method but does not replace this new-image experiment.
- [x] Mark independent forest/urban reference areas in the test image for
  quantitative comparison. Leave uncertain and other-cover pixels unlabelled;
  explain that a two-class classifier still forces a choice on valid pixels
  outside its intended two-class domain.
- [x] Save the exact training pixel indices/features/labels and evaluation
  masks, plus an input manifest. If sampling is needed for kNN runtime, choose
  a fixed, documented sample per class and use it for Bayes too.
- [x] Define one common feature transformation: preferably training-set
  z-score scaling per band, with safe handling of a zero standard deviation.
  Save the training means/scales and apply them unchanged to every test image.
  Fit neither scaling nor model choices on test pixels or reference labels.

Completion evidence: notebook data overview, labelled ROI figures, separate test
preview, saved masks/sample selection and source manifest under `lab3/data/`.

## 2. Plot class histograms and estimate parametric PDFs — required

Completed in section 2 of [analysis.ipynb](analysis.ipynb), using the exact saved
512 samples per class. Three figures show shared-bin histograms with Gaussian
marginals, finer forest distributions and band correlations. Full Gaussian MLE
parameters, regularized covariances in raw/scaled units, equal priors and fixed
training scaling are saved in [results/bayes/gaussian_model.json](results/bayes/gaussian_model.json).
Written observations describe class separation, skewness, tail mismatch, joint
covariance, prior assumptions and regularization sensitivity.
[Verification](results/bayes/verification.json) checks all nine histogram exports,
six normalized Gaussian marginals, raw/scaled transformations, a hand-calculated
MLE example and saved model/table consistency. Test pixels/labels were not used
for fitting or tuning.
All 13 code cells passed a fresh-kernel run with saved outputs; the three new
figures and observations were reviewed. See
[results/bayes/phase2_review.json](results/bayes/phase2_review.json).

- [x] Plot forest and urban training histograms for each input band: six
  distributions, arranged for direct class comparison. Use common bins and
  axes per band, density normalization, units, legends and sample counts.
- [x] Check that each full-range histogram integrates to approximately one
  as `sum(density * bin_width)`; disclose display limits or clipping.
- [x] Inspect peaks, spread, skewness, tails and class overlap. Explain why
  the selected features may separate forest from urban and where they overlap.
- [x] State the baseline model `x | c ~ N(mu_c, Sigma_c)`. Estimate each
  class mean vector and full 3 × 3 maximum-likelihood covariance from its
  training samples, using denominator `N_c`. Report the numerical parameters
  and whether they are expressed in raw or standardized feature units.
- [x] Add a small, recorded diagonal regularization to each covariance for
  stable scoring. Report its value and any sensitivity worth discussing.
- [x] Overlay fitted univariate Gaussian marginal PDFs on the corresponding
  histograms in matching units. Explain that these plots show marginals;
  the classifier uses the joint PDF and cross-band covariance.
- [x] Assess the fit honestly. Discuss multimodal urban materials and
  departures from Gaussian shape; a Gaussian is a modelling assumption,
  not something established merely by calculating a mean and variance.
- [x] State class priors explicitly. Start with equal priors for the balanced
  lab comparison; explain that deliberately balanced ROI samples do not
  establish the real forest/urban proportions of the test image.

Completion evidence: histogram/PDF figure, written fit assessment, parameter
table and saved model/scaling settings under `lab3/results/bayes/`.

## 3. Implement and apply the Bayes classifier from scratch — required

- [ ] Explain Bayes' rule and the equal-misclassification-cost decision:
  choose the class maximizing `log p(x | c) + log P(c)`; the common evidence
  term `p(x)` cancels when comparing classes.
- [ ] Write the multivariate Gaussian log-score explicitly, including the
  covariance determinant and quadratic term. Use a stable solve/factorization
  and log determinant; avoid multiplying tiny PDFs or unnecessary inversion.
- [ ] Implement fitting and prediction with NumPy primitives. Define a
  deterministic rule for equal class scores and verify finite model scores.
- [ ] Pass **every valid pixel** of the new image through the classifier,
  reshape predictions to the original grid and preserve invalid pixels as
  nodata. Label the map clearly: **forest = red, urban = blue**; display
  invalid pixels neutrally or transparently.
- [ ] Show the original test image, Bayes class map and a useful detail crop
  with a legend and acquisition information. Save a lossless class raster
  with georeferencing and a separate coloured visualization.
- [ ] Explain the actual observed successes and errors, including shadows,
  mixed pixels, vegetation within cities and differences between acquisitions.
  Distinguish visual interpretation from reference-label evidence.

Completion evidence: documented decision rule, implementation, saved Bayes map
and written interpretation in the main notebook.

## 4. Implement kNN from scratch and run k = 1, 3, 5 — required

- [ ] Explain the algorithm step by step: store training feature vectors and
  labels; transform a query pixel with training settings; compute Euclidean
  distances; select the k nearest training samples; predict by majority vote.
- [ ] Implement distance computation, neighbour selection and voting directly.
  Use floating-point features so subtraction/squaring cannot overflow an
  integer image dtype. Require at least five training samples.
- [ ] Define deterministic handling of equal distances at the kth boundary,
  for example distance followed by saved training index. Odd k avoids a
  two-class vote tie but does not remove equal-distance neighbour ambiguity.
- [ ] Process query pixels in manageable batches rather than storing the
  full image-by-training distance matrix. Keep the prediction mathematically
  equivalent to exhaustive neighbour search and document the sample size.
- [ ] Classify the same valid test pixels for **k = 1, 3 and 5**, holding the
  training set, feature transformation, distance metric and masks fixed.
- [ ] Save all three class rasters and visualizations with the same
  red/blue legend and nodata treatment as Bayes. Show original, k = 1,
  k = 3 and k = 5 side by side, plus matched enlarged details.
- [ ] Explain that neighbours are selected in spectral feature space.
  Increasing k does not directly apply a spatial smoothing filter.

Completion evidence: readable algorithm explanation, scratch implementation,
three prediction maps and the comparison figure under `lab3/results/knn/`.

## 5. Compare results and explain the effect of k — required discussion

The assignment requires results and observations; independent reference labels
and the following compact measurements make those observations testable.

- [ ] Evaluate Bayes and all three kNN variants on the **same independently
  labelled test pixels**, excluding nodata and unlabelled/other-cover pixels.
  Report evaluated support per class and the limitation of partial references.
- [ ] Present a confusion matrix with named axes and a small table containing
  accuracy, per-class precision/recall and balanced accuracy or macro F1.
  Define how undefined metric denominators are handled.
- [ ] Measure the fraction of valid test pixels whose predictions change
  from k = 1 to 3, 3 to 5 and 1 to 5; show a disagreement map or matched detail
  crop where it helps explain the changes.
- [ ] Answer explicitly: what changed at **k = 1, k = 3 and k = 5** in the
  observed class maps and metrics? Discuss isolated decisions, spectral
  outliers, preservation/loss of small features and majority-class effects
  only where supported by the results. Report small/no differences honestly.
- [ ] Explain the expected trade-off: k = 1 is more sensitive to individual
  samples; larger k aggregates more evidence but can bias ambiguous pixels.
  Do not claim accuracy must improve monotonically with k.
- [ ] Compare Bayes' parametric PDF assumptions with kNN's sample-based
  decisions, including training/prediction cost and feature scaling. Record
  training sample counts and, if comparing speed, comparable prediction timing.
- [ ] Discuss limitations: manual label uncertainty, within-class diversity,
  spatial dependence, two-class coverage, acquisition/illumination differences
  and the risk that separate cities/forests encode scene-specific conditions.
  Conclusions apply to the evaluated imagery and reference areas.

Completion evidence: comparison table, confusion matrices, k-change evidence
and explicit answers to both assignment questions in the notebook.

## 6. Final notebook and submission checks — required

- [ ] Check assignment coverage: training images/ROIs, histograms, fitted PDFs
  and parameters, Bayes test and red/blue result, all kNN steps, three k maps
  and evidence-based comments on their differences.
- [ ] Verify essential algorithm behaviour with small hand-checkable examples:
  Gaussian score/decision against manual calculation, known nearest neighbours
  and votes for each k, equal-distance handling and batch-size invariance.
  Check disjoint training/evaluation indices and unchanged test preprocessing.
- [ ] Check mask/raster shape, class IDs, valid-pixel coverage, georeferencing,
  colour legend, finite scores and confusion-matrix totals. Verify plots are
  readable, comparable and consistent with saved arrays/metrics.
- [ ] Restart the notebook kernel, run all cells in order using the locked
  project environment, resolve errors/hidden state and save all outputs.
- [ ] Save seeds, ROI definitions, input/output hashes, model parameters,
  feature scaling, sample counts, k values, implementation settings and package
  versions. Keep original data unchanged and temporary credentials out of files.
- [ ] Finish `lab3/README.md` with exact run instructions, provenance,
  attribution, output locations and clearly stated reuse of earlier lab files.
  Reproduction with saved inputs should not need network access or credentials.
- [ ] Review the notebook's written answers and figures; update this checklist
  only when each item has corresponding evidence. Record verification results
  and remaining limitations in `lab3/SUBMISSION_REVIEW.md`.
- [ ] Prepare a portable submission ZIP following the earlier labs' convention.
  Include the notebook, required local inputs (including referenced earlier-lab
  files), masks, implementation, outputs, docs and dependency lock files. Check
  it from a relocated copy; omit environments and caches.

Completion evidence: executed notebook, finished README and submission review,
checked-off checklist and a verified portable submission archive.
