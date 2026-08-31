# Project 01 — In-Cylinder Flow Reconstruction & Prediction

**Status: 🟡 Days 1-5 technical work done — physical-observation notes pending final write-up**

## Engineering problem
In-cylinder flow structure (tumble/vortex behavior, cycle-to-cycle variability) measured via PIV directly affects mixture formation and combustion quality. Can an ML model reconstruct or predict flow fields from partial/available data cheaply enough to be useful alongside experiment/CFD?

## Objective and outcomes

**Objective:** show that reduced-order/ML techniques applied to real experimental in-cylinder PIV data can (1) compress high-dimensional turbulent flow into a tractable representation, (2) interpolate/predict flow state from sparse operating-condition data, and (3) reconstruct fields from incomplete measurements - the three capabilities that make ML actually useful in a CFD/test workflow, not just an accuracy demo.

**Outcome for this Phase 1 subset:** a working, honest, real-data pipeline covering all three, with results reported including where they're weak (Stage 4) rather than only where they're strong. This is not a production-ready predictive model - the dataset is intentionally 1/80th the size of the full one. What it does demonstrate is the right *method* for each capability, validated well enough to know which parts would benefit most from scaling up. See "Phase 2" below for what that scale-up looks like and why each piece matters in a real CFD workflow, not just as a portfolio exercise.

## Physics baseline
- PIV plane, crank-angle-resolved U/V velocity components
- Velocity magnitude, vector/quiver structure, vorticity
- Cycle-to-cycle variability in engine flow

## Dataset
- Oxford EngineBench: https://eng.ox.ac.uk/tpsrg/research/enginebench
- Kaggle (LSP Small): https://www.kaggle.com/datasets/samueljbaker/enginebench-lsp-small
- Full dataset: https://www.kaggle.com/datasets/samueljbaker/enginebench
- Reference code: https://github.com/sambkr/EngineBench

## Method
Progression: data/physics understanding → cycle-to-cycle variability (EDA) → POD/PCA reduced-order representation → regression from crank angle to POD coefficients → sparse-to-full field reconstruction (gappy POD). All on the real EngineBench LSP Small dataset (`r1300_p40`, 5 crank angles, 1041 cycles each), downloaded via the Kaggle API.

## Result
- **Cycle-to-cycle variability is highest early in the cycle and decays through it**: field-averaged std of velocity magnitude drops from 5.36 (cad090) to 1.83 (cad270) across the same 1041 cycles.
- **POD compression is slow for this flow**: 265 of 1041 possible modes are needed for 90% energy, 717 for 99% - this is a turbulence-dominated flow, not one with a few dominant coherent structures. Even so, 5 modes alone already recover the large-scale vortex/tumble pattern visually, even at 68% pointwise error - large-scale structure and fine-scale turbulent energy are not the same thing.
- **Crank-angle-only regression is weak, and data-starved rather than method-limited**: a global linear regression across all 4 training crank angles gives 77.4% relative error, barely beating a naive "reuse the nearest known crank angle" baseline (78.5%). Switching to local interpolation between just the two *nearest* known crank angles drops this to 55.2% - a real improvement that also shows the bottleneck is sparse sampling (5 points total), not the modeling approach. See "Genuine limitations" below.
- **Sparse-to-full reconstruction (gappy POD) works reasonably for large-scale structure**: with 30% of spatial points masked, gap-point reconstruction error is 46.3%, and the recovered field visually preserves the dominant vortex/tumble pattern while losing fine turbulent detail.

| Stage | Task | Key result | Figure |
|---|---|---|---|
| 1 | Data/physics exploration | Loaded real HDF5 structure, plotted U/V/magnitude/quiver/vorticity | `results/day1_piv_snapshot.png` |
| 2 | Cycle-to-cycle variability (EDA) | Variability std: 5.36 (cad090) → 1.83 (cad270) | `results/day2_cycle_variability.png` |
| 3 | POD/PCA | 265/1041 modes for 90% energy; 5 modes already show large-scale structure | `results/day3_pod_energy.png`, `results/day3_pod_reconstruction.png` |
| 4 | Crank angle → POD coeffs regression | Global regression 77.4% error vs 78.5% naive baseline; local nearest-neighbor interpolation improves this to 55.2% | `results/day4_regression_result.png` |
| 5 | Sparse/gappy-POD reconstruction | 46.3% error on masked points, large-scale structure recovered | `results/day5_sparse_reconstruction.png` |

Full process, code and day-by-day learning notes: `day1_explore_piv.py` … `day5_sparse_reconstruction.py` and the matching `day*_learning_notes.md` files in this folder.

## Error analysis
Error is reported per-task above rather than as one number, since each day answers a different question (reconstruction fidelity vs. predictive generalization vs. gap-filling accuracy). Day 4 is the clearest limitation: with only 5 crank angles in this small dataset, a global regression cannot learn the real nonlinear crank-angle → flow relationship (77.4% error). Local interpolation between the two nearest known angles does meaningfully better (55.2%) precisely because it doesn't try to extrapolate a trend across widely-spaced points - which is itself evidence the bottleneck is sampling density, not the modeling approach. See "Genuine limitations" below for what full-scale data would need.

## Engineering conclusion
This flow is turbulence-dominated rather than structure-dominated: POD needs hundreds of modes for high energy capture, yet the large-scale tumble/vortex pattern is visually recoverable from just a handful of modes or from 70% of the spatial points. That distinction - compressibility of large-scale structure vs. fine-scale turbulent energy - is the throughline across Days 3-5, and it's also why a crank-angle-only regression (Day 4) underperforms: it can only ever predict the smooth/large-scale part, not the turbulent variability characterized in Day 2.

## Genuine limitations, and what full-scale data would need

This Phase 1 project intentionally used the Kaggle "LSP Small" subset: **one operating condition (1300 RPM, 40 kPa), 5 crank angles, 390 MB**, downloaded and run on a personal machine. Two honest limitations follow directly from that:

- **Day 4's regression is data-starved, not a modeling failure.** 4 training points can't teach a model the real (nonlinear) crank-angle → flow relationship - local interpolation between the two nearest known angles already beat a "smarter" global regression (55.2% vs 77.4% relative error), which is itself evidence that the bottleneck is sampling density, not method choice.
- **The full EngineBench dataset is 31 GB** (`LSP.h5` alone - the file this 390 MB subset came from - is 13.3 GB, and contains many more test points/operating conditions). That's not something to pull down to a laptop; it needs to run where it lives. The right environment is a **Kaggle Notebook** running directly against the attached dataset (free tier: ~20 GB persistent disk, GPU quota) rather than a local download - this is also why the original project plan specified Kaggle as the compute layer, not just a data source.

**Open question for anyone reading this** (genuinely - not rhetorical): if you're working through AI/ML-for-CFD problems on a personal machine rather than a company cluster, how are you handling datasets in the tens-of-GB range? Cloud notebooks, a home GPU box, university compute, something else? This project will move to the full dataset in a Kaggle Notebook next - if you've solved this tradeoff differently, I'd like to hear about it in [Discussions](../../discussions).

## Phase 2 queue: full dataset, and why each piece matters in real CFD work

Not in scope for this Phase 1 subset - queued for the full 31 GB EngineBench dataset, run in a Kaggle Notebook rather than locally (see "Genuine limitations" above). Each item below is a real capability used in industrial CFD/test programs, not just a harder ML exercise:

| Backlog item | Why it matters in a real CFD/test workflow |
|---|---|
| Multi-step crank-angle forecasting (current state → future state) | Basis for real-time, control-oriented engine models - relevant to knock prediction and combustion control, not just offline analysis |
| ConvLSTM / temporal U-Net surrogate | Replaces expensive transient CFD runs during design-space screening - the actual reason surrogate modeling exists in industry |
| Physics-aware losses (divergence/vorticity/circulation) | A surrogate that violates conservation laws isn't trustworthy for engineering decisions - this is what separates a demo model from one an engineer would actually sign off on |
| Cycle-variability / anomaly detection | Directly usable as a combustion-instability diagnostic in engine test cells, not just a research metric |
| Uncertainty quantification | Any prediction feeding an engineering decision needs a confidence bound - a point estimate alone isn't enough to act on |
| Sparse-sensor placement optimization | Answers "where should we actually put PIV/pressure sensors" - a real, direct cost lever on physical test programs |
| CFD-to-experiment domain transfer | The sim-to-real gap is one of CFD's persistent open problems - ML-assisted correction against test data is an active industry direction, not a solved one |

This table itself is meant to answer a specific question: **not "can this predict well," but "which of these, if built out, would an engineer actually reach for."**

## Source attribution
Dataset and reference implementation: Samuel J. Baker, Michael A. Hobley, Isabel Scherl, Xiaohang Fang, Felix C. P. Leach, Martin H. Davy (Oxford TPSRG).

Citation: Baker et al., "EngineBench: Flow Reconstruction in the Transparent Combustion Chamber III Optical Engine," arXiv:2406.03325, 2024. Dataset DOI: 10.34740/KAGGLE/DS/5000332.

Required acknowledgment (per the source repository): "The TCC engine work has been funded by General Motors through the General Motors University of Michigan Automotive Cooperative Research Laboratory, Engine Systems Division."

This project is an independent learning exercise using the public dataset and referencing the public data-loading code structure, not affiliated with the original authors.
