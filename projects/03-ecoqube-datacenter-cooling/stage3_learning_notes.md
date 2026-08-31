# Stage 3 - surrogate-based cooling optimization

Grid search over the surrogate's actual trained operating range (5 features, 8 levels each), constrained to predicted max rack temp <= 33.8C.

Result: recommended setting achieves predicted Tmax=33.71C (essentially the same as the historical average of 33.71C) at a cooling-power proxy of 3.67, a **23.5% reduction** versus the historical mean cooling power (4.79).

## Physical observations (my own words)

1. The tradeoff plot isn't a clean monotonic curve - at nearly every cooling-power level there are BOTH feasible (green... actually red in my color scheme, feasible=safe) and infeasible points spanning a wide temperature range. That tells me max rack temp isn't driven by cooling power alone - the combination of settings matters, not just "more cooling = always cooler," which is exactly why a multi-feature surrogate search is more useful here than tuning one knob at a time.
2. The recommended setting sits right at the edge of the feasible region for its power level, not comfortably in the middle - which makes sense for a minimum-power search: the optimizer is deliberately finding the cheapest setting that just barely satisfies the safety constraint, not the safest setting overall. Worth stating plainly so this doesn't read as "found a magic free lunch."
3. A 23.5% cooling-power reduction for the same predicted temperature is a real, useful finding IF the surrogate is trustworthy - which ties directly back to Stage 2's honesty caveat: this search stayed inside the surrogate's actual observed range, so I'm not claiming it found something genuinely new, just the best of what was already measured.

## What I'm still unsure about

- Whether "cooling power proxy" (coolOutput + fan speed/100) is a physically meaningful combined unit or just a convenient scalar I made up for this exercise - a real deployment would need an actual energy-cost model, not this proxy.
