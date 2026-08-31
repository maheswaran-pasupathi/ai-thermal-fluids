# Project 02 — Spray & Combustion ML

**Status: 🔵 Stage 1 complete, Stages 2-3 in progress**

![Lift-off length vs. ambient temperature and other conditions](results/stage1_liftoff_vs_inputs.png)

## Engineering problem
Diesel spray lift-off length is a key combustion KPI - it governs mixing before ignition and strongly affects soot formation. Can I predict it from injection/ambient conditions with a model that's also physically interpretable, not just accurate?

## Objective and outcomes
I set out to show that ML on real ECN combustion data can:
- Turn a messy, multi-study experimental table into a clean, usable dataset
- Predict a real combustion KPI (lift-off length) from ambient/injection conditions
- Explain *why* the model predicts what it does (SHAP), and check that against known combustion physics

Stage 1 is done: 948 raw rows cleaned to 336 usable rows, with a real data-cleaning bug caught and fixed along the way (see notes). Stages 2-3 (regression + SHAP) are next.

## Physics baseline
Ambient O2 concentration, ambient temperature, ambient density, injection pressure, and nozzle orifice diameter all influence ignition delay and flame stabilization (lift-off length) in a diesel spray.

## Dataset
Engine Combustion Network (ECN), Sandia National Laboratories - the open experimental-data table behind their Diesel Spray Combustion search tool. Freely downloadable, no account required.
- Data search tool: https://ecn.sandia.gov/diesel-spray-combustion/experimental-data-search/
- Direct CSV: https://ecn.sandia.gov/databases/dieseldata.csv
- Column definitions: https://ecn.sandia.gov/diesel-spray-combustion/experimental-data-search/definitions/

## Method
Stage 1: clean the raw CSV (948 rows), select lift-off length as the target (best coverage of the four candidate KPIs), verify physical sensibility with EDA. Stage 2: linear baseline → Random Forest / XGBoost regression. Stage 3: SHAP feature importance, checked against known combustion physics.

## Result (Stage 1)
- 336 of 948 rows have every core input (O2%, Ta, density, injection pressure, orifice diameter) and the target present after cleaning
- Ambient temperature shows the clearest relationship with lift-off length - it drops sharply as Ta rises, matching known combustion chemistry
- Injection pressure and orifice diameter cluster around a handful of common test conditions rather than varying continuously - a real constraint on what Stage 2's model can learn, not something I'm glossing over

| Stage | Task | Key result | Figure |
|---|---|---|---|
| 1 | Data cleaning + EDA | 336/948 clean rows; temperature is the strongest visible driver | <img src="results/stage1_liftoff_vs_inputs.png" width="160"> |

Full code and my stage-by-stage notes: `stage1_data_cleaning.py` and the matching `stage*_learning_notes.md` files.

**Appendix (exploratory, not a numbered stage):** I also tried validating the tabulated lift-off values against their source OH* chemiluminescence images directly, independent of the CSV. Real result (r=0.95 on 7 of 19 images) and a real open question I haven't resolved (my threshold wasn't the documented ECN measurement definition) - see `appendix_image_validation.py` and its notes.

## Error analysis
Not applicable yet - Stage 1 is data cleaning, no model trained. Will report per-stage from Stage 2 onward, same standard as Project 01.

## Genuine limitations
This table pools many different studies/rigs rather than one controlled sweep, so injection pressure and orifice diameter are heavily clustered around common test conditions (e.g. ~150 MPa, 0.09/0.18mm). A regression model trained on this will generalize better within those clusters than outside them - I'll report that honestly in Stage 2 rather than only citing an aggregate error number.

## How to reproduce
1. `pip install -r ../../requirements.txt`
2. Download the data directly, no account needed: `curl -o ecn_dieseldata.csv https://ecn.sandia.gov/databases/dieseldata.csv`
3. Run `stage1_data_cleaning.py` (`# %%` cell blocks).

## Source attribution
Data: Engine Combustion Network (ECN), Sandia National Laboratories, and the contributing research institutions whose experiments populate this table (attribution per-row via the table's `refs`/`fileBaseName` columns on the ECN site).

Please cite the ECN and the original experimental paper(s) for any use of this data beyond a learning exercise - see https://ecn.sandia.gov/ for citation guidance.

Independent learning exercise using the public ECN dataset - not affiliated with Sandia National Laboratories or the ECN.
