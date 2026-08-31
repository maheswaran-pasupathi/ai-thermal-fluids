# Project 03 — Data Center Cooling Surrogate

**Status: ⚪ Planned**

## Engineering problem
Build a thermal surrogate for small data-center cooling that could support control/optimization decisions faster than running full CFD.

## Physics baseline
Server heat load, airflow/cooling configuration, and resulting temperature distribution; transferable to electronics and battery-pack cooling problems.

## Dataset
ECO-Qube (EU project) — AI-augmented cooling for small data centres.
- https://cordis.europa.eu/project/id/956059
- CFD/experimental data: https://zenodo.org/records/7035829
- OpenFOAM validation: https://zenodo.org/records/6336674
- Server power resource: https://zenodo.org/records/11372221

## Method
Regression-based thermal surrogate, with an optimization/control framing (minimize cooling power subject to a max-temperature constraint).

## Result
_To be added._

## Source attribution
Data credited to the ECO-Qube EU project and contributing partners (CORDIS project 956059).
