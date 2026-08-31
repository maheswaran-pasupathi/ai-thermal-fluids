# Project 06 — Thermal Digital Twin (Flagship)

**Status: ⚪ Planned**

## Engineering problem
Predict transient temperature (and optionally velocity/pressure) field evolution from geometry, boundary conditions and current state — a fast surrogate ("digital twin") for conjugate heat transfer problems. Directly transferable to battery cooling, electronics cooling, cold plates, HVAC, and data-center thermal management.

## Physics baseline
Transient conjugate heat transfer: geometry, boundary conditions, current temperature/flow state → future field.

## Dataset
"Learning Transient Convective Heat Transfer with Geometry-Aware World Models" (2026).
- https://zenodo.org/records/18889216
- https://doi.org/10.5281/zenodo.18889216
- https://arxiv.org/abs/2601.22086

## Method
First model: U-Net for field prediction. Later comparison against FNO/DeepONet if time allows. Target presentation layer: a lightweight interactive app (e.g. Streamlit) taking geometry/heat-load/flow-rate inputs and returning predicted temperature evolution, max temperature, and inference time vs CFD runtime.

## Result
_To be added._

## Source attribution
Dataset credited to the original paper's authors (arXiv:2601.22086 / Zenodo 18889216).
