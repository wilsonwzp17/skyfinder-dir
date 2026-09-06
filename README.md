# SkyFinder-DIR

Predicting temperature from a single outdoor webcam image, adapting the
[Deep Imbalanced Regression](https://github.com/YyzHarry/imbalanced-regression) code to SkyFinder.

The write-up is `report/REPORT.md`.

## Data

SkyFinder ships 53 camera archives. Six did not fetch from the project-page route this pipeline
uses, leaving 47 whose images can be paired with the table's temperature rows; I use the twelve
smallest, about 400 MB together.

| | |
|---|---|
| Target | temperature, degrees C |
| Temperature range, these twelve cameras | -13.2 to 39.4 |
| Binning domain | -30 to 50 C, 81 bins of 1 C |
| Cameras | 12 |
| Usable images | 12,962 |
| Split, in distribution | 10,489 train / 1,322 val / 1,151 test |
| Split, camera held out | four folds, whole cameras removed |

Two things to know about the labels. The temperature comes from a nearby weather observation rather
than a reading at capture time, a median 88.6 minutes away, with the filename clock and the
observation clock both in UTC. And the image count narrows in three
steps: the twelve archives hold 13,544 JPEGs, 13,011 of which have a temperature row in the released
table, and 49 of those are truncated and are dropped, leaving 12,962. The truncated files are listed
in `data/corrupt_images.csv`; `data/build_cache.py` prints these three counts when it builds the
cache, and `data/camera_manifest.csv` carries the per-camera matched and cached counts that sum to
13,011 and 12,962.

## Setup

```bash
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
PY=../.venv/bin/python
$PY data/download.py
```

Python 3.9; every result records the interpreter and torch versions it was run under, 3.9.23 and
2.8.0, and the torch version is pinned in `requirements.txt`. Then `./reproduce.sh` builds the labels, the decode cache and both split regimes, runs the
controls and the audits, trains, and writes the tables. The splits and results that are already
committed are skipped, because each result records the sha256 of the split it came from. Training
itself is `run_all.sh`, which `reproduce.sh` calls.

## Running

```bash
$PY dir/train.py --splits data/splits_iid.csv --arm erm --seed 0 --out results/iid_erm_s0.json
$PY dir/train.py --splits data/splits_iid.csv --arm lds --seed 0 --reweight sqrt_inv \
    --out results/iid_lds_s0.json
$PY dir/train.py --splits data/splits_cd0.csv --arm erm --seed 0 --out results/cd0_erm_s0.json
```

`--arm` is one of `erm`, `sqinv`, `lds`, `fds`, `lds_fds`; `sqinv` reweights without the LDS kernel
and `lds` is SQINV + LDS. `--basis` picks the shot-region definition, marginal target bins or joint
(bin, camera) groups; every stored result uses `bin`, the DIR definition.

## What came from where

- `dir/vendor/` is the reference DIR implementation at commit `a6fdc45`, the `imdb-wiki-dir`
  folder, with every edit listed in `dir/vendor/PATCHES.md`.
- The LDS weights in `dir/train.py` follow the reference `_prepare_weights` step for step, with a
  floor-based bin index in place of the `int()` lookup so that signed temperatures stay in range.
- Kernel size 5 and sigma 2 follow the reference README examples and the 2026 SkyFinder benchmark.
  FDS momentum 0.9, updating from epoch 0 and smoothing from epoch 1 follow the pinned trainer's
  defaults. L1 loss, the input normalisation, the shot thresholds 100 and 20 and checkpoint
  selection by validation MAE follow the reference implementation. The one-degree temperature bins
  are my adaptation of the reference's one-year age granularity.
- ResNet-18, the joint group definition `g = (temperature bin, camera)` and SGD with momentum 0.9 and weight decay 1e-4 come from the 2026 paper that benchmarks SkyFinder, scaled down to a laptop as the report's Setup states.
- The temperature label is the closest weather observation to each frame, as the SkyFinder paper
  states; `audit/label_timing.py` measures how close.
- Local additions are the signed-temperature binning, the twelve-camera subset and both splits, the choice to run the reference's `--reweight sqrt_inv` without `--lds` as a separate SQINV arm so that the smoothing can be isolated, the no-pixel and leakage controls, the label-timing audit,
  the FDS diagnostics, the paired analysis, and the result and report validation scripts.

## How the study fits together

1. Data: `data/download.py` fetches the twelve archives and the table, `data/build_labels.py`
   pairs images with temperatures, `data/build_cache.py` decodes every image once, and
   `data/build_splits.py` builds both split regimes.
2. Signed temperatures: `common.py` holds the binning, the two shot-region definitions, the metrics
   and the table shape; `dir/porting_bugs.py` runs the three things that break when the reference code meets a signed target and the weight-concentration comparison behind the report's implementation section.
3. Training: `dir/train.py` is the loop, the LDS weights and the FDS wiring over the vendored
   `dir/vendor/`, one result file per run; `run_all.sh` runs the whole programme.
4. What the metadata alone predicts, and what the split leaks: `baselines/trivial.py` holds three predictors that never see an image, the bar the image model has to clear, and the camera-only mean behind the interaction sentence under the report's no-pixel controls;
   `probes/leakage.py` runs two structural checks and four measurements on a split.
5. Camera held out: the four `data/splits_cd*.csv` folds, trained by the same loop.
6. What the label measures: `audit/label_timing.py`, how far the label is from the photograph.
7. Reading the results: `analysis/paired_arms.py` for paired per-image differences, per-seed sign
   counts and camera-clustered standard errors; `dir/test_fds_last_epoch.py` asserts the two
   inherited FDS state behaviours the report's implementation section describes; `check_results.py` checks every result
   against a manifest and rejects any FDS-bearing result that lacks the frozen-BatchNorm flag or
   the occupied bucket range; `fill_report.py` writes the two ablation tables into the report.

## Results

MAE, mean and standard deviation across three seeds. Shot regions over marginal target bins.

| arm | Overall<br><sub>n=1151</sub> | Many<br><sub>n=862</sub> | Medium<br><sub>n=163</sub> | Few<br><sub>n=77</sub> | Zero<br><sub>n=49</sub> |
|---|---|---|---|---|---|
| ERM | 3.80 ±0.09 | 3.16 ±0.06 | 4.34 ±0.10 | 7.30 ±0.29 | 7.61 ±0.44 |
| SQINV | 3.73 ±0.03 | 3.23 ±0.04 | 3.99 ±0.04 | 6.54 ±0.19 | 7.22 ±0.03 |
| SQINV + LDS | 3.79 ±0.07 | 3.36 ±0.07 | 3.93 ±0.07 | 6.35 ±0.28 | 6.83 ±0.12 |
| FDS | 3.73 ±0.07 | 3.19 ±0.07 | 4.10 ±0.23 | 6.68 ±0.36 | 7.33 ±0.14 |
| SQINV + LDS + FDS | 3.76 ±0.07 | 3.42 ±0.06 | 3.61 ±0.06 | 6.01 ±0.24 | 6.70 ±0.29 |

Three predictors that use no image at all, on the same test set: 4.54 keyed on camera, month and
three-hour block; 7.99 on month and block; 10.15 for the training mean.

## What the stored results do and do not record

Each result records the arm, seed, split path and hash, shot basis, reweighting, epochs, learning
rate, batch size, Python and torch versions, checkpoint epoch, residuals at full precision, and the
FDS fields the checker validates; `check_results.py` rejects any result whose split has changed or
whose settings differ from its manifest. Other fixed choices, the optimiser momentum, weight decay,
schedule, kernel parameters, preprocessing and cache paths, live in the committed trainer rather
than in each result. Nothing binds a result to the pixels: the decode cache and its
index are gitignored, so two caches with the same index satisfy every stored hash and could give
different residuals. When `reproduce.sh` rebuilds the cache it compares the new index against the
tracked `data/labels_cached.csv` and stops on any difference, which fixes the row order and not the
decoded images. `dir/train.py` as committed wrote every file in `results/`; no result records a hash
of the code or the command line, so that binding rests on this sentence.

## Licence

Code is MIT. SkyFinder is on Zenodo under CC BY 4.0, record 5884485; the dataset asks to be cited
as Mihail, Workman, Bessinger and Jacobs, "Sky Segmentation in the Wild: An Empirical Study", WACV
2016.

## Sources

- Delving into Deep Imbalanced Regression: https://proceedings.mlr.press/v139/yang21m.html
- The paper that benchmarks SkyFinder: https://arxiv.org/abs/2606.01723
- SkyFinder: https://zenodo.org/records/5884485
