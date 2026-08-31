# Stage 1 - ECN spray/combustion: data cleaning notes

Dataset: ECN diesel spray/combustion table (`ecn_dieseldata.csv`), 948 raw rows, cleaned to 336 rows with all 5 core inputs (ambient O2%, ambient temperature, ambient density, injection pressure, orifice diameter) and lift-off length present.

## Physical observations (my own words)

1. Ambient temperature is clearly the strongest driver of lift-off length in this data - it drops sharply as Ta increases, which lines up with combustion chemistry: hotter ambient air means faster ignition kinetics, so the flame stabilizes closer to the nozzle.
2. Ambient O2 concentration shows a real but much weaker pattern - the scatter is a lot noisier than temperature, which makes sense since I'm pooling many different studies/rigs together in this table, not one controlled sweep.
3. Injection pressure and orifice diameter both cluster hard around a few common test conditions (150 MPa, 0.09/0.18mm) rather than being swept continuously - I'll need to keep that in mind for Stage 2, since a regression model will see a lot of near-duplicate rows at those clusters and comparatively few examples elsewhere.

## Data-cleaning gotcha (real, not hypothetical)

Naive `pd.to_numeric()` on the raw columns silently dropped good data: many cells embed an HTML annotation link straight after the number (e.g. `"141.1;<a href=...>fuel pressure vs time</a>"`), so a plain numeric conversion turned those into NaN. That took my clean row count from an expected ~338 down to 34 before I caught it and switched to extracting the leading numeric token with a regex instead. Worth remembering for any messy real-world CSV, not just this one.

## ML/data concepts - what I now understand

- Missing-value encoding varies by source: this table uses "-" for missing, not blank cells, and I had to explicitly tell pandas that (`na_values=["-"]`).
- "Non-missing" isn't the same as "clean": a cell can have a non-null value that's still not directly usable (the HTML-annotation case above).
- Pooled multi-study data isn't the same as a designed experiment - the clustering around a few common test conditions is a real constraint on what a model trained on this table can learn, not something I should ignore going into Stage 2.

## What I'm still unsure about

-
