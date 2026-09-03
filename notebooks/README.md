# Notebooks

Learning-in-public notebooks. Each has a `.ipynb` (rendered, with outputs) and a
paired `.py` (jupytext percent format) which is the version to diff and review.

## `fundamentals.ipynb`

Small self-contained numerical experiments, no external data: Biot/Fourier and
when the lumped-capacitance assumption is legal; transient integration schemes
(forward/backward Euler, RK4) and their stability limit and accuracy order; 1-D
conduction with generation vs analytic plus a grid-convergence check. Each result
is checked against a closed-form solution.

## `battery_rul.ipynb`

Open-data investigation: capacity fade / remaining-useful-life on the Kaggle
*Battery Remaining Useful Life (RUL)* dataset
(`ignaciovinuales/battery-remaining-useful-life-rul`, CC0-1.0). One engineering
question, the full source -> data-quality -> characterisation -> baseline model
-> cell-wise validation -> residual physics chain. `Cycle_Index` is deliberately
excluded (predicting cycles-left from cycle-number is circular here).

Data: on Kaggle, attach the dataset through the UI. Locally:

```
mkdir -p data
kaggle datasets download -d ignaciovinuales/battery-remaining-useful-life-rul -p data --unzip
```

The dataset file is not committed here.

## `modelica_basics.ipynb`

OpenModelica from the compiler only (`omc` + `.mos` script -> CSV -> matplotlib,
no GUI). Mass-spring-damper, RC step response, and a lumped thermal body, each
built from Modelica Standard Library components and checked against its
closed-form solution. Needs OpenModelica on the machine.

## Reproducing

```
pip install -r ../requirements.txt jupytext nbclient ipykernel
jupytext --to notebook <name>.py
jupyter execute --inplace <name>.ipynb
```
