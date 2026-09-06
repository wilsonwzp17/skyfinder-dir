# Adapting DIR to SkyFinder

Zhaopeng (Wilson) Wu, September 2026

I used this project to answer two questions.

1. Does Label Distribution Smoothing improve temperature prediction beyond square-root inverse
   frequency weighting?
2. Do the results transfer when the test cameras are not seen during training?

I also tested predictors that do not use images. This helped show how much of the task can be
solved from camera and time information alone.

## Data and setup

SkyFinder contains 53 camera archives. Six did not download from the original project-page route
used by my pipeline. From the remaining 47, I selected the twelve smallest archives, about 400 MB
in total, so I could run the experiments on a laptop.

After matching images with temperature rows and removing unreadable files, the dataset has 12,962
images. The twelve cameras contain between 175 and 1,805 images each. Their temperatures range
from -13.2 to 39.4 degrees Celsius. The full released table ranges from -27.2 to 50.0 degrees.
Since the cameras were chosen by archive size, the results describe this subset rather than all of
SkyFinder.

LDS and FDS come from the Deep Imbalanced Regression paper and its code. From the 2026 SkyFinder
benchmark paper I used the ResNet-18 backbone, joint camera-group definition, and SGD settings as
starting points. That paper trains for 400 epochs on an A40 GPU with batch size 256 and learning
rate 0.2. My version uses 112 by 112 images, batch size 64, and 40 epochs. I scaled the learning
rate to 0.05 and kept SGD, momentum 0.9, weight decay 1e-4, and cosine annealing.

The model starts from random weights and uses L1 loss. SQINV, SQINV + LDS, and SQINV + LDS + FDS
use square-root inverse frequency weights. ERM and FDS do not. I use one-degree bins and Gaussian
LDS and FDS kernels with size 5 and sigma 2. The best checkpoint is selected by validation MAE.

I use two splits. The first is camera-overlapping and group-capped. Images are grouped by
temperature bin and camera. Each group contributes up to three test images and four validation
images, with test filled first. I chose these caps to roughly match the benchmark's split
proportions, but this is not the benchmark split, and the repository name `iid` does not mean a
normal random split.

The second split holds out cameras. The twelve cameras are divided into four test folds, with
three test cameras per fold. Each fold also uses two separate cameras for validation. The model is
then tested on cameras it has never seen.

## Results

### No-pixel controls

I tested three simple predictors before interpreting the image models.

*LOOKUP* uses camera, month, and three-hour time block. *CALENDAR* uses only month and time block,
so it works on unseen cameras. *GLOBAL* always predicts the mean training temperature.

| Setting | LOOKUP | CALENDAR | GLOBAL |
|---|---|---|---|
| Camera-overlapping split | 4.54 | 7.99 | 10.15 |
| Camera held out, mean of four folds | 7.67 | 6.98 | 7.67 |
| Camera held out, range across folds | 5.98 to 10.24 | 5.45 to 8.97 | 5.98 to 10.24 |

On the camera-overlapping split, LOOKUP improves over GLOBAL by 55.3 percent. CALENDAR improves it
by 21.3 percent. Camera and calendar information interact, so the gap between LOOKUP and CALENDAR
is not a clean measure of the value of camera identity. With cameras held out, LOOKUP has no entry
for a test camera, so it falls back to GLOBAL and becomes identical to it.

ERM reaches 3.80 MAE, compared with 4.54 for LOOKUP and 7.99 for CALENDAR. However, predicting
each test label from the nearest training frame of the same camera reaches 3.35 without using
pixels. A past-only version reaches 3.92. The image model beats the coarse controls but not the
nearest-frame control. The camera-overlapping split therefore contains strong time information.

### What the temperature label represents

Each image is matched with a nearby weather observation. The weather timestamp usually differs
from the capture time in the filename. The METAR time and AMOS filename time are both in UTC or
GMT. Reading the filename as local time would raise the median gap to 207 minutes, so I use the
documented GMT interpretation. Of the 12,962 images, 11,300 have METAR timestamps that my parser
can read. The other 1,662 use AAXX reports. For the parseable images, the weather observation is a
median of 88.6 minutes from the photograph.

For 3,186 images, the capture time falls between two observations no more than three hours apart.
Interpolating between them gives a difference of about 0.6 degrees when the released label is one
endpoint and 1.9 degrees when it is not. These are two different subsets rather than bounds, and
the released data contain no true capture-time temperature, so the label is best treated as a
nearby weather observation rather than a capture-time measurement.

### DIR results on the camera-overlapping split

Each cell below shows MAE as mean ± standard deviation across three seeds. The value after the
slash is the geometric mean of the errors, averaged across the three seeds. Shot regions are based
on marginal temperature bins.

<!-- BEGIN:iid_ablation -->
| Arm | Overall, n=1,151 | Many, n=862 | Medium, n=163 | Few, n=77 | Zero, n=49 |
|---|---|---|---|---|---|
| ERM | 3.80 ±0.09 / 2.44 | 3.16 ±0.06 / 2.05 | 4.34 ±0.10 / 3.02 | 7.30 ±0.29 / 5.92 | 7.61 ±0.44 / 6.33 |
| SQINV | 3.73 ±0.03 / 2.39 | 3.23 ±0.04 / 2.09 | 3.99 ±0.04 / 2.61 | 6.54 ±0.19 / 5.10 | 7.22 ±0.03 / 5.88 |
| SQINV + LDS | 3.79 ±0.07 / 2.45 | 3.36 ±0.07 / 2.17 | 3.93 ±0.07 / 2.67 | 6.35 ±0.28 / 4.80 | 6.83 ±0.12 / 5.36 |
| FDS | 3.73 ±0.07 / 2.37 | 3.19 ±0.07 / 2.04 | 4.10 ±0.23 / 2.76 | 6.68 ±0.36 / 5.16 | 7.33 ±0.14 / 5.90 |
| SQINV + LDS + FDS | 3.76 ±0.07 / 2.43 | 3.42 ±0.06 / 2.22 | 3.61 ±0.06 / 2.28 | 6.01 ±0.24 / 4.60 | 6.70 ±0.29 / 5.36 |
<!-- END:iid_ablation -->

SQINV + LDS against ERM changes both weighting and smoothing. SQINV alone separates them. Because
every arm evaluates the same 1,151 test images, I compare absolute errors image by image rather
than only comparing seed means. Negative means the second method has lower error.

| Comparison | Many | Medium | Few | Zero |
|---|---|---|---|---|
| SQINV minus ERM | +0.06, 0 of 3 seeds lower | -0.35, 3 of 3 | -0.76, 3 of 3 | -0.39, 2 of 3 |
| SQINV + LDS minus SQINV | +0.14, 0 of 3 | -0.07, 2 of 3 | -0.19, 3 of 3 | -0.39, 3 of 3 |
| SQINV + LDS minus ERM | +0.20, 0 of 3 | -0.41, 3 of 3 | -0.95, 3 of 3 | -0.78, 3 of 3 |

Square-root inverse weighting produces most of the improvement in the medium and few-shot regions.
Adding LDS gives smaller gains there and another 0.39-degree gain in zero-shot bins. Both steps
raise error in the many-shot region. Overall MAE is 3.80 for ERM, 3.73 for SQINV, and 3.79 for
SQINV + LDS. Among these three arms, SQINV gives the best overall result. LDS moves more of the
improvement toward sparse bins but loses overall accuracy compared with SQINV alone.

All three seeds agree on the direction in ten of the twelve comparison cells. The two exceptions
are SQINV in the zero-shot region and the added LDS effect in the medium-shot region.
Camera-clustered standard errors for SQINV + LDS against ERM are 0.06, 0.19, 0.53, and 0.26
degrees across the four regions.

The benchmark paper defines groups by both temperature bin and camera. Under that definition, 233
test images are zero-shot. Only 49 are also zero-shot under the marginal temperature-bin
definition used for the main DIR analysis. I do not interpret the FDS rows because of the
implementation behavior described below.

### Camera-held-out results

The camera-held-out results use one run per arm and fold, so I report each fold rather than
averaging away the variation.

<!-- BEGIN:cd_ablation -->
| Fold | Test images | CALENDAR | GLOBAL | ERM | SQINV | SQINV + LDS | SQINV + LDS + FDS |
|---|---|---|---|---|---|---|---|
| 0 | 3,629 | 5.45 | 6.74 | 6.61 | 6.74 | 8.41 | 8.41 |
| 1 | 2,740 | 5.60 | 7.72 | 8.93 | 8.43 | 9.83 | 9.42 |
| 2 | 4,094 | 7.91 | 5.98 | 6.12 | 6.33 | 8.81 | 8.58 |
| 3 | 2,499 | 8.97 | 10.24 | 9.26 | 9.18 | 8.72 | 8.40 |
<!-- END:cd_ablation -->

CALENDAR beats each image model on three of four folds. No image model beats it on more than one
fold. The good camera-overlapping results therefore do not transfer reliably to new cameras.
SQINV changes ERM by +0.13, -0.51, +0.21, and -0.07 degrees across the four folds. Adding LDS to
SQINV changes MAE by +1.67, +1.40, +2.48, and -0.46 degrees. Its effect depends strongly on the
held-out cameras.

In fold 0, LDS improves the medium, few, and zero-shot regions while raising overall MAE. In folds
1 and 2, it raises error in every populated region. Fold 3 is mixed. The four single-seed folds
show variation across cameras, but they are not enough to rank the methods.

Four of the sixteen runs select epoch 0 or 1 as the best checkpoint. In fold 0, SQINV + LDS and
SQINV + LDS + FDS have identical residuals because both select epoch 1, before FDS changes their
predictions. Only three cameras are ever used for validation, so checkpoint selection is also
uneven across folds.

## What I learned from the implementation

The original DIR code assumes a non-negative target. Its weight code builds keys starting from
zero, so a temperature such as -13.2 tries to access `value_dict[-13]` and fails. Adding a constant
shift does not fully solve this. Python's `int()` truncates toward zero, while temperature bins
need `floor()`. The two methods disagree for negative values and make the bin around zero too
wide. I use one shared `bin_index()` based on `floor()`.

FDS also needs integer bin indices instead of raw temperatures. Passing raw temperatures creates
417 distinct labels but updates only 40 buckets. All 906 negative labels enter bucket zero. I
corrected two errors from my initial adaptation before running the final 32 experiments. First, I
used raw inverse-frequency weights instead of `sqrt_inv`. This produced a weight ratio of 303 and
placed about one fifth of the total weight on one percent of the samples. Second, FDS included 21
empty buckets below the observed training range. The final code limits FDS to the occupied bins.

I also found that the FDS feature-collection pass changed BatchNorm statistics. The model must be
in training mode to return its features, and `no_grad()` does not stop BatchNorm updates. I tested
this with an FDS control that collects statistics but never smooths. It did not match ERM until I
froze BatchNorm during feature collection. After that change, the control became bit-identical to
same-seed ERM. The result checker now tests this before accepting any FDS run.

Two behaviors remain in the pinned FDS code. The saved raw statistics share storage with the
current statistics, so the raw and smoothed values can come from different epochs. Smoothing also
modifies the feature tensor in place, so later statistics are collected from already-smoothed
features. I left these behaviors unchanged and treat all FDS results as exploratory.

## Next steps

The first next step is to use more cameras, repeat the camera-held-out runs across seeds, and
rotate the validation cameras. I would also use a day-blocked split because 95.0 percent of the
current test images share a camera and day with training data. A third experiment would rebuild
labels from observations available before capture. Finally, I would compare the pinned FDS
behavior with a version that uses raw and smoothed statistics from the same epoch and collects
unsmoothed features.

## References

Yang, Y., Zha, K., Chen, Y.-C., Wang, H., and Katabi, D. "Delving into Deep Imbalanced
Regression." ICML 2021. <https://proceedings.mlr.press/v139/yang21m.html>

Reference DIR implementation, pinned at commit `a6fdc45`.
<https://github.com/YyzHarry/imbalanced-regression>

Xu, G., Li, J., Wang, H., and Yang, Y. "Shortcut to Nowhere: Demystifying Deep Spurious
Regression." arXiv:2606.01723. <https://arxiv.org/abs/2606.01723>

Mihail, R. P., Workman, S., Bessinger, Z., and Jacobs, N. "Sky Segmentation in the Wild: An
Empirical Study." WACV 2016. SkyFinder dataset, CC BY 4.0.
<https://doi.org/10.5281/zenodo.5884485>

NOAA JetStream. "Z-time (Coordinated Universal Time)." <https://www.noaa.gov/jetstream/time>

AMOS dataset page, archived April 14, 2019. The page documents filename timestamps as GMT.
<http://web.archive.org/web/20190414211057/http://amos.cse.wustl.edu/dataset>
