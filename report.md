# Point-Supervised Semantic Segmentation — Technical Report

## Problem

Normal segmentation training needs a full pixel-level mask for every training
image, which is expensive to produce. This task asks for a way to train a
segmentation model using only a handful of labeled points per image instead
of a full mask, and to measure how well that actually works.

## Method

**Partial cross-entropy (pCE) loss.** I implemented a loss function that
computes the standard per-pixel cross-entropy everywhere, then multiplies it
by a binary mask marking which pixels actually have a label. Unlabeled
pixels get multiplied by zero and contribute nothing to the loss or the
gradient — the model simply never "hears" anything about them. I verified
this directly: with a full mask the loss is numerically identical to plain
cross-entropy, and with a sparse mask, the gradient at an unlabeled pixel is
exactly 0.0 while labeled pixels get a nonzero gradient.

**Point simulation.** Since I didn't have real point-annotated data, I
simulated it: given a full ground-truth mask, I randomly sample N pixels per
class and discard the rest, keeping only those N as "known." Everything
else is marked unlabeled. This lets me control exactly how sparse the
supervision is and measure the effect.

**Model.** A U-Net with a ResNet34 encoder pretrained on ImageNet. The
encoder progressively downsamples the image while extracting features; the
decoder (which I built myself, ResNet doesn't include one) upsamples back to
full resolution. Skip connections carry fine-grained spatial detail directly
from each encoder stage to its matching decoder stage, so the output isn't
blurry. Using a pretrained encoder matters a lot here specifically because
supervision is so sparse — starting from ImageNet weights means the model
already knows general visual features (edges, textures) and only has to
adapt them to the remote-sensing domain, rather than learning everything
from a handful of labeled pixels.

**Evaluation.** mIoU (mean Intersection-over-Union), computed per class
against the *full* ground-truth mask — the full mask is never used for
training, only kept aside to measure true accuracy.

## Implementation

Code is split by responsibility:
- `losses.py` — partial cross-entropy loss
- `point_sampling.py` — simulates sparse point labels from a full mask
- `model.py` — ResNet34-based U-Net
- `dataset.py` — dataset wrapper + mIoU metric
- `train.py` — training/validation loop
- `experiment.py` — runs the points-per-class comparison

Each piece was unit-tested individually before wiring them together (e.g.
confirming pCE collapses to plain cross-entropy when the mask is full,
confirming the model's output resolution matches its input, confirming
mIoU = 1.0 for a perfect prediction) and then validated end-to-end with a
synthetic-data smoke test before moving to real data.

## Experiment: effect of points-per-class

Everything (images, model, training length) was held fixed; only the number
of labeled points per class was varied.

**Synthetic data (sanity check, no label noise):**

| points/class | mIoU |
|---|---|
| 3  | 0.125 |
| 10 | 0.244 |
| 30 | 0.340 |

A clean, expected trend — more labeled points, steadily better accuracy.
This confirmed the core mechanism works before testing on anything messier.

**Real data (LandCover.ai, 5 classes, GPU training, ImageNet-pretrained
encoder):**

| points/class | best mIoU |
|---|---|
| 5  | 0.363 |
| 20 | 0.382 |
| 50 | 0.341 |

## Discussion

The synthetic result shows the exact trend you'd hypothesize: more
supervision, better accuracy, with no surprises, because the synthetic data
has clean, well-separated classes and no real-world noise.

The real-data result is messier, and I think that's worth discussing
honestly rather than glossing over. 20 points slightly beat 5, but 50 didn't
beat 20. Two likely reasons:

1. **Validation set size.** I used 30 validation images, which makes the
   per-epoch mIoU a fairly noisy estimate — I saw single-epoch swings as
   large as 0.05 to 0.44 within one training run, just from batch
   composition. "Best mIoU across 8 epochs" is somewhat sensitive to that
   noise, especially comparing across only three runs.
2. **Random point selection isn't necessarily informative point selection.**
   Points are currently sampled uniformly at random. Going from 5 to 20
   points likely adds genuinely new information. Going from 20 to 50 may
   mostly add redundant points (e.g. more pixels from the interior of a
   large uniform field) that don't teach the model much it didn't already
   know, while also adding more noisy/ambiguous pixels near class
   boundaries.

If I extended this, the two things I'd try first are: a larger validation
set to get a more stable mIoU estimate, and boundary-aware point sampling
(biasing point selection toward edges between classes, where labels are
most informative) instead of uniform random sampling across the whole
image.

## Limitations

- Training was short (8-10 epochs) given time constraints; results would
  likely improve with longer training.
- Validation set (30 images) is small, which limits how much weight should
  be put on small differences between runs.
- Point sampling is uniform random rather than strategic.

## What I'd do with more time

- Boundary-aware or uncertainty-based point sampling instead of uniform
  random
- A semi-supervised term that also uses the ~90%+ unlabeled pixels (e.g. an
  entropy-minimization regularizer), rather than discarding them entirely
- Test-time augmentation and ensembling, as mentioned in the original slide,
  to reduce the per-epoch noise seen in the mIoU curves
