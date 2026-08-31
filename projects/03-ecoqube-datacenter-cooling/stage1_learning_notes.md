# Stage 1 - ECO-Qube: CFD vs. experiment, and live sensor exploration

Real data: exhaust-air CFD numerical profiles + experimental measurements (both cooling designs), plus 721-sample rack-back temperature and cooling-unit sensor logs for each design.

## Physical observations (my own words)

1. The CFD and experimental exhaust temperature profiles follow the same shape - both show a clear hot band around 1.5-1.9m height where exhaust air is warmest, and both agree the retrofitted design runs hotter at the top of that band (up to ~35C vs ~33C for the previous design). But there's a consistent offset between the two curves rather than a tight overlay - the CFD systematically undershoots the highest experimental points. That's a real validation gap, not something to gloss over.
2. The live sensor data (rack-back temperature by U position) shows the same qualitative pattern independently - hot spots cluster around U34-40, cooler air around U20-26 - which is a second, independent confirmation of where the thermal problem actually is in the cabinet, not just a CFD artifact.
3. The retrofitted design's sensor profile tracks almost identically to the previous design's in most places, with the clearest difference showing up at the coolest points (U20-26) rather than the hottest ones - suggests the retrofit changed airflow distribution at the bottom of the rack more than it changed the hot-spot severity at the top.

## Data-cleaning gotcha (real, not hypothetical)

The raw sensor logs aren't valid JSON despite looking like it - they're Python dict literals with single quotes, one per line (`{'timestamp': [list of {sensor: value}]}`), which needed `ast.literal_eval` per line rather than `json.loads`. The CFD CSV files also weren't consistent with each other - the previous-design file had 3 columns (with a trailing blank), the retrofitted-design file had only 2 - caught by an immediate column-count mismatch error, not a silent wrong-column bug.

## ML/data concepts - what I now understand

- CFD-vs-experiment validation isn't optional context here, it's the actual Day 1 task - before trusting any CFD-derived surrogate later, I need to know how well the CFD itself tracks reality.
- Two independent measurement sources (CFD/experimental exhaust profile vs. live rack sensors) agreeing on the same qualitative pattern is much stronger evidence than either alone - that's the same principle as Project 2's SHAP-vs-physics check, applied to raw data instead of a trained model.

## Stage 1b - rendering the actual 3D CFD field, not just a 1D profile

I went further than the exhaust-height profile and loaded the real solved OpenFOAM case (previous design, solved timestep) with pyvista, then rendered temperature slices directly from the 3D field.

1. The side view shows exactly what the exhaust profile in Part A implied but couldn't show directly: cool air (blue) sitting low near the rack intakes, hot air (red/orange) collecting at the ceiling - real hot-air-rises physics, not just a number trend on a chart.
2. The plan view shows a clear, localized hotspot right at the rack exhaust boundary, surrounded by cooler room air everywhere else - this is the kind of thing a scalar Tmax number completely loses. Two different racks can have the same Tmax but a very different hotspot footprint, and that matters for where you'd actually place cooling.
3. This confirms the sensor-derived and exhaust-profile findings from Part A/B independently a third way - three different data sources (CFD field, CFD/experimental exhaust profile, live rack sensors) all agree on where the heat actually is. That's a strong basis for the Stage 2 surrogate model to build on.

## What I'm still unsure about

-
