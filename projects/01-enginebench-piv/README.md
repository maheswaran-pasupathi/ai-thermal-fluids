# Project 01 — In-Cylinder Flow Reconstruction & Prediction

**Status: 🔵 In progress — Day 1**

## Engineering problem
In-cylinder flow structure (tumble/vortex behavior, cycle-to-cycle variability) measured via PIV directly affects mixture formation and combustion quality. Can an ML model reconstruct or predict flow fields from partial/available data cheaply enough to be useful alongside experiment/CFD?

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
Progression: data/physics understanding → POD/PCA reduced-order representation → regression from crank-angle/condition to POD coefficients → reconstructed field → (stretch) sparse-to-full field reconstruction.

## Result
_To be added as the project progresses._

## Error analysis
_To be added._

## Engineering conclusion
_To be added._

## Phase 2 backlog (not in scope yet)
Multi-step crank-angle forecasting, ConvLSTM/temporal U-Net, physics-aware losses (divergence/vorticity/circulation), cycle-variability/anomaly detection, uncertainty quantification, sparse-sensor placement optimization, CFD-to-experiment domain transfer.

## Source attribution
Dataset and reference implementation: Samuel J. Baker, Michael A. Hobley, Isabel Scherl, Xiaohang Fang, Felix C. P. Leach, Martin H. Davy (Oxford TPSRG).

Citation: Baker et al., "EngineBench: Flow Reconstruction in the Transparent Combustion Chamber III Optical Engine," arXiv:2406.03325, 2024. Dataset DOI: 10.34740/KAGGLE/DS/5000332.

Required acknowledgment (per the source repository): "The TCC engine work has been funded by General Motors through the General Motors University of Michigan Automotive Cooperative Research Laboratory, Engine Systems Division."

This project is an independent learning exercise using the public dataset and referencing the public data-loading code structure, not affiliated with the original authors.
