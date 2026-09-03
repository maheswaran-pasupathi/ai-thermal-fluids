# Notebooks

Learning-in-public notebooks. Each has a `.ipynb` (rendered, with outputs) and a
paired `.py` (jupytext percent format) which is the version to diff and review.

## `engineering_lab.ipynb` — cumulative engineering lab

One running notebook, not a new one per day. Two parts:

- **Part A – fundamentals:** self-contained numerical experiments (Biot/Fourier
  and the lumped-capacitance assumption, transient integration schemes and their
  stability/accuracy order, 1-D conduction with generation vs analytic +
  grid convergence).
- **Part B – open-data investigation 1:** capacity fade / remaining-useful-life
  on the Kaggle *Battery Remaining Useful Life (RUL)* dataset
  (`ignaciovinuales/battery-remaining-useful-life-rul`, CC0-1.0). Cell-wise
  train/test split, physically-signed coefficients, honest held-out error.
- **Part C:** method notes on ARC-tracing battery thermal-runaway modelling,
  tied to the `BatteryTR` Modelica library in `vehicle-systems-engineering`.

Data: on Kaggle, attach the dataset through the UI. Locally:

```
mkdir -p data
kaggle datasets download -d ignaciovinuales/battery-remaining-useful-life-rul -p data --unzip
```

The dataset file is not committed here.

## `modelica_basics.ipynb` — system-modelling stream, M1

OpenModelica from the compiler only (`omc` + `.mos` script → CSV → matplotlib,
no GUI). Mass–spring–damper, RC step response, and a lumped thermal body, each
built from Modelica Standard Library components and checked against its
closed-form solution. Needs OpenModelica on the machine (`omc` on PATH or at
`D:\OpenModelica\bin\omc.exe`).

## Reproducing

```
pip install -r ../requirements.txt jupytext nbclient ipykernel
jupytext --to notebook <name>.py
jupyter execute --inplace <name>.ipynb
```
