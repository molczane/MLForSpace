# Lab 2 completion checklist

Based on the supplied assignment screenshot and the current repository. All
required questions must be answered in the submitted notebook, including written
observations and conclusions alongside the code and plots.

## Current starting point

- [x] Create `lab2/`.
- [x] Locate the supplied `Basic_opt.ipynb`: optimization examples followed by
  the optional compressive-sensing exercise. It does not yet contain the required
  Sentinel histogram or bit-manipulation experiments.
- [x] Identify reusable Sentinel-2 inputs in `../lab1/data/sentinel2/`:
  Kraków (urban), Białowieża (natural forest), and Baltic Sea (additional natural
  region). Each has three-band DN and reflectance GeoTIFFs, previews and metadata.

## 1. Prepare the analysis notebook and inputs — required

Completed in [analysis.ipynb](analysis.ipynb); see [README.md](README.md) and
[data/manifest.json](data/manifest.json) for inputs, provenance and verification.

- [x] Create a main notebook in `lab2/` with sections matching assignment
  questions 1 and 2, plus question 3 if attempted. Keep the supplied tutorial as a
  reference.
- [x] Use Kraków and Białowieża as the urban/natural pair (selected by the user;
  no Baltic region included).
- [x] Reuse at least three Sentinel-2 bands for **each** selected region. The
  existing RGB GeoTIFFs can supply B04, B03 and B02; verify band order in source
  metadata. Use band values for the histogram analysis, retaining display PNGs
  for previews or the explicitly documented LSB experiment.
- [x] Record the input paths, source scene IDs, dates, band names, dimensions,
  pixel spacing, footprints, units and preprocessing in the notebook or linked
  metadata. Document any reuse of lab 1 files.
- [x] Obtain Sentinel-1 imagery covering the same geographic footprints. Choose
  acquisition dates close to the Sentinel-2 observations where practical and
  record any temporal differences.
- [x] Choose and document the Sentinel-1 product, available polarizations
  (e.g. VV/VH), calibration/terrain-correction status, and whether values are
  amplitude, intensity/backscatter in linear units, or dB. Use a clearly defined
  representation consistently. The assignment explicitly requires three bands
  for Sentinel-2; do not invent three Sentinel-1 optical-style bands.
- [x] Crop Sentinel-1 to the same regions and check coverage, valid pixels,
  geolocation and image quality. Explain any differences in grid/resolution or
  resampling; pixel-perfect alignment is not necessary for regional histograms.
- [x] Define valid-pixel masks and treatment of clouds, shadows, nodata and
  invalid SAR values. State whether the crops mix land-cover types.
- [x] Confirm dependencies and a reproducible way to run the notebook using the
  project environment. Save inputs or enough acquisition metadata to reproduce
  them without embedding credentials.

## 2. Probability distributions / histograms — required (question 1)

Completed in section 2 of [analysis.ipynb](analysis.ipynb), using the labelled
previews in section 1. Four histogram figures, 14 sets of bin counts/densities,
summary statistics and normalization checks are saved in
[results/histograms/](results/histograms/). Written observations cover both sensors,
band/polarization differences, urban–natural contrasts and interpretation limits.

- [x] Show labelled Sentinel-2 previews of the urban and natural regions.
- [x] Plot a histogram for each of the three selected Sentinel-2 bands in each
  region: at least **six distributions** for the minimum two-region setup.
- [x] Define whether plots show counts, probability mass, or estimated density.
  For density, normalize so the sum of density × bin width is approximately 1;
  for probability mass, normalize so bin probabilities sum to 1.
- [x] Label band/polarization, region, units, sample count and histogram axes.
  Use common bins and axis limits for comparable datasets; disclose any clipping
  or log scaling. Do not compare optical and SAR raw numeric scales as if their
  units were equivalent.
- [x] Write observations about the Sentinel-2 distributions: band differences,
  spread, skewness, peaks and urban-versus-natural differences. Relate the
  observations to the displayed terrain and scene conditions without claiming
  terrain is the only cause.
- [x] Show Sentinel-1 previews and repeat the histogram analysis and written
  observations for the same regions and selected polarizations.
- [x] Discuss how Sentinel-1 and Sentinel-2 distributions differ, with appropriate
  caveats about sensor physics, SAR speckle, units, preprocessing and dates.

## 3. Replace the two least significant bits — required (question 2)

The screenshot labels this section “Stegenography”; the requested experiment is
random replacement of two LSBs, with no message payload required.

Completed in section 3 of [analysis.ipynb](analysis.ipynb) for the original
uint16 Sentinel-2 B04 bands of **both Kraków and Białowieża**. Image comparisons,
fine/coarse histograms, bit/error diagnostics, lossless rasters, metrics and
reproduction metadata are saved in [results/lsb/](results/lsb/). All three
written questions are answered with evidence and detection limitations.

- [x] Select at least one image used in question 1 and document its region,
  sensor, band/channel and integer bit depth. A single integer band is sufficient
  for the minimum experiment.
- [x] If the chosen data are floating point, explicitly define an integer
  quantization for this experiment and retain that quantized image as the
  unchanged comparison baseline. Do not apply integer bit operations directly
  to physical floating-point values. **Not applicable here:** the experiment
  uses original integer DN directly, with no new quantization.
- [x] Generate an independent random integer in `{0, 1, 2, 3}` for every pixel,
  using a recorded random seed. Replace, rather than add to, the two low bits:
  `modified = (original & clear_low_two_bits_mask) | random_values`, using a
  mask appropriate for the integer dtype. If using multiple channels, document
  the randomization per channel.
- [x] Check that dimensions and dtype are preserved, upper bits are unchanged,
  and the absolute change is at most 3 integer levels. Save the result losslessly
  if writing an image file.
- [x] Display original and modified images side by side with identical display
  limits, plus an enlarged crop and a labelled difference image.
- [x] Answer **“Does it affect visual quality?”** using actual inspection;
  support it with change statistics such as changed-pixel fraction, maximum
  absolute error and MAE or PSNR. Express errors relative to the chosen bit depth.
- [x] Plot original and modified histograms/PDF estimates with identical bins,
  normalization and axes. Include a fine-bin comparison or difference plot so
  broad bins do not hide changes among groups of four consecutive levels.
- [x] Answer **“Do you see any difference?”** based on the results. Inspect the
  two-bit value frequencies or adjacent-level groups as supporting diagnostics
  if useful; distinguish these diagnostics from the required full histogram.
- [x] Answer **“Can this detect satellite-image manipulation?”** Discuss what
  the observed distribution changes suggest and the limitations: histogram
  differences alone do not prove tampering, and absence of a visible difference
  does not establish authenticity. Separate access to a known original from
  detection using only a suspect image.

## 4. Compressive sensing — optional (question 3)

Attempted and completed in section 4 of [analysis.ipynb](analysis.ipynb).
The supplied signal, sampling and three optimization formulations were run
with a fixed seed, retaining their solver defaults and recording unsuccessful
termination honestly. A separate corrected basis-pursuit comparison and exactly
sparse positive control explain the model/solver limitations. Three figures,
metrics, arrays and run settings are saved under
[results/compressive_sensing/](results/compressive_sensing/). The original
`Basic_opt.ipynb` remains unchanged; its unused `skopt` import/install is omitted
from the reproducible main-notebook version, and SciPy is explicitly locked.

- [x] Decide whether to attempt this optional section; explicitly mark it as
  omitted if not included.
- [x] If attempted, run the compressive-sensing section of `Basic_opt.ipynb`
  reproducibly, saving plots of the original signal, sparse measurements,
  transform coefficients and reconstructions.
- [x] Resolve dependencies in the project environment: the tutorial imports
  SciPy and `skopt`, while `scikit-optimize` appears unused. Avoid relying on
  an unrecorded notebook `pip install` for reproducibility.
- [x] Check optimizer convergence and reconstruction error, and explain failed
  or incomplete runs honestly. Inspect the tutorial's claims against outputs:
  the signal need not have exactly two nonzero DCT coefficients, and minimizing
  measurement error alone does not enforce sparsity.
- [x] If repairing the optional example, document changes and use a suitable
  sparse-recovery formulation; explain that L1 promotes sparse coefficients.
- [x] Write where compressive sensing may be useful in Earth observation
  (EO), such as reducing acquisition/transmission needs under a suitable sparse
  model, together with assumptions and limitations. Do not treat this toy signal
  as evidence that arbitrary satellite images can be recovered from few samples.

## 5. Final notebook and submission checks — required

Completed: all **17 code cells** passed in order after a kernel restart, with
saved outputs. Twelve figures were reviewed, and the exported numerical results
were checked against source/saved arrays. The artifact checks also passed on an
isolated relocated copy with network calls disabled. See
[SUBMISSION_REVIEW.md](SUBMISSION_REVIEW.md),
[results/notebook_review.json](results/notebook_review.json) and
[results/submission_checks.json](results/submission_checks.json).
The portable ZIP includes the referenced lab 1 files and has a per-file SHA-256
inventory; build/rebuild it with `uv run python lab2/submission.py --package`.

- [x] Ensure every required assignment question has an explicit written answer
  in the main notebook, with conclusions supported by its own plots/results.
- [x] Restart the kernel and run the main notebook from top to bottom; save
  outputs and resolve errors, hidden state and missing local paths.
- [x] Verify figures are readable and use consistent labels, units and scales.
- [x] Include source/provenance notes, preprocessing decisions, random seeds,
  input references and environment/run instructions under `lab2/`.
- [x] Check the notebook and required local inputs/results are included in the
  submission and can be opened without private credentials. Keep large source
  downloads outside version control where appropriate, with reproducible access
  instructions.
