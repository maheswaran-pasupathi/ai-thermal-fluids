# Appendix - image-processing validation of tabulated lift-off length

**Status: exploratory side-track, not part of the main Stage 1-3 plan.** Kept because the result and the debugging process are both real and worth showing, but demoted from a numbered stage since it raised a question I haven't resolved (see below).

19 rows in the ECN table link directly to their source OH* chemiluminescence image. I extracted lift-off length myself from the raw images (intensity thresholding along the jet axis) and checked it against the tabulated mm value - independent of the CSV entirely.

## Open question I have not resolved
My threshold (15% of the image's max intensity) was a guess, not the documented ECN measurement definition - I checked the ECN definitions page and it doesn't specify the exact threshold used for lift-off length from chemiluminescence images (it does specify 50%-of-steady-signal for a *different* metric, ignition delay). So the r=0.95 correlation shows my threshold tracks the same physical trend as the tabulated values, but I can't currently claim it reproduces the researchers' exact measurement procedure. Worth resolving before treating this as a validated method rather than a promising exploratory result.

## Physical observations (my own words)

1. On the 7 images from Sandia's own raw-frame format, my extracted pixel position correlates at r=0.95 with the tabulated mm value - strong enough that I'm confident the threshold approach is finding the same physical feature (the OH* onset) that the original researchers measured, not something else.
2. The other 12 images turned out to be a completely different format from a different rig - post-processed plots with axis labels, tick marks and a white border baked in, rather than a raw camera frame. I didn't expect that going in; I found it by noticing my code returned an identical wrong answer (pixel 0) for every single one of them, which is a much stronger signal of a systematic bug than a scatter of random errors would be.
3. Chasing that bug taught me more than the correlation number did: the border artifact wasn't just a simple margin, it was a border line -> anti-aliasing halo -> tick marks bleeding inward, each one requiring a bigger crop before the next one showed up. That's a real lesson about post-processed/rendered images versus raw sensor output - rendering artifacts don't have a single clean edge.

## Data-cleaning gotcha (real, not hypothetical)

The border pixels in the second image format are pure white (1.0), which becomes the *global max* of the image - so a threshold defined as a fraction of the max (0.15 x max) gets satisfied instantly by the border itself, not the real spray signal. This is the same class of bug as Stage 1's HTML-annotation issue: a value that looks structurally fine (a number, a pixel) but silently isn't the thing I actually wanted.

## Why I didn't force a number for the second format

I could have kept shrinking the crop margin until *some* correlation appeared, but that's fitting noise to a target, not validating anything. The honest call: the raw Sandia format validates cleanly (r=0.95, n=7), and the post-processed format needs real axis-detection or OCR-based calibration to handle properly - out of scope for this stage, logged as a real limitation instead of quietly worked around.

## ML/data concepts - what I now understand

- A systematic identical failure (all 12 images returning exactly 0) is a much stronger debugging signal than scattered errors - it means "wrong code," not "noisy data."
- Thresholding relative to an image's own max intensity is fragile if there's any chance of a rendering artifact brighter than the real signal.
- Grouping by a detected property (border presence) rather than by filename pattern is more robust - the filename convention happened to line up with the format split here, but I didn't rely on it.

## What I'm still unsure about

-
