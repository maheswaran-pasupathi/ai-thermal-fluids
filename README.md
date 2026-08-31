# AI for Thermal & Fluid Engineering

**CFD & Thermal Engineering × Scientific Machine Learning**

Battery & Electro-Thermal Management • Conjugate Heat Transfer • Combustion • Vehicle Aerodynamics • Surrogate Modeling • Neural Operators

Built by [Maheswaran Pasupathi](https://www.linkedin.com/in/srimahes) — a senior CFD/thermal simulation engineer (8+ years; STAR-CCM+, CONVERGE, GT-SUITE, Python + Java-macro automation) who has driven 40–50% simulation-turnaround reduction through workflow automation, now applying AI/ML — surrogate modeling, reduced-order models, neural operators — to the same physics he works in daily: battery thermal runaway, electro-thermal busbar behavior, conjugate heat transfer, IC combustion, and vehicle aerodynamics.

[Connect on LinkedIn](https://www.linkedin.com/in/srimahes) to discuss CFD, thermal engineering, or applying AI/ML to physics problems.

This repo is a public, in-progress portfolio — projects are added and updated as they're built, not dumped in finished. Every project favors **physical interpretation over model-accuracy claims**, states clearly what stage it's at, and ships a runnable notebook rather than plots alone.

## Why this repo exists

Most "AI for CFD" content online is either pure ML with no engineering grounding, or pure CFD with no ML. This portfolio is an attempt to bridge that gap from the CFD-engineer side — using real physics datasets, explaining the engineering meaning of every result, and being explicit about where a model's predictions can and can't be trusted.

## Portfolio

| # | Project | Physics | AI/ML Method | Status |
|---|---|---|---|---|
| 01 | [In-Cylinder Flow Reconstruction](projects/01-enginebench-piv/) | Engine PIV flow, tumble/vortex structure, cycle variability | POD/PCA → regression → reconstruction | 🔵 In progress |
| 02 | [Spray & Combustion ML](projects/02-ecn-spray-combustion/) | Ignition delay, spray penetration, lift-off length | Random Forest / XGBoost + SHAP | ⚪ Planned |
| 03 | [Data Center Cooling Surrogate](projects/03-ecoqube-datacenter-cooling/) | Thermal surrogate + cooling control/optimization | Regression + optimization | ⚪ Planned |
| 04 | [Vehicle Aerodynamics AI](projects/04-drivaernet-aero/) | Drag coefficient prediction from geometry | XGBoost + SHAP explainability | ⚪ Planned |
| 05 | [Electronics Hotspot Prediction](projects/05-electronics-thermal-cnn/) | Power/heat-source layout → temperature map | CNN | ⚪ Planned |
| 06 | [Thermal Digital Twin (flagship)](projects/06-transient-cht-digital-twin/) | Transient conjugate heat transfer, battery/electronics cooling transfer | U-Net → FNO/DeepONet | ⚪ Planned |
| 07 | [Scientific ML / Neural Operators](projects/07-cfdbench-neural-operator/) | Canonical CFD PDE solution fields | Fourier Neural Operator | ⚪ Planned |

Each project card follows the same structure: **Engineering problem → Physics baseline → Dataset → Method → Result → Error analysis → Engineering conclusion**, with one visual result and full source attribution to the original dataset/paper.

## Community

This repo is also meant as a shared learning space for CFD engineers — experienced or just starting out — who want to pick up AI/ML without losing the physics grounding that makes CFD work trustworthy.

- **Discussions** are open — ask questions, suggest datasets, share your own approach to any of the problems above, or point out where a result looks physically wrong.
- **Issues** are open for dataset suggestions, method suggestions, or spotted errors.
- See [CONTRIBUTING.md](CONTRIBUTING.md) if you'd like to contribute a notebook, fix, or project card of your own.

If you're a CFD/thermal engineer curious about ML, or an ML person curious about CFD, you're in the right place.

## Datasets used (with attribution)

All datasets are public/research datasets, used with attribution to their original authors. Links and citations are in each project's own README. No proprietary or employer data is used anywhere in this repository.

## License

Code in this repository is released under the [MIT License](LICENSE). Datasets remain under their original licenses — see each project folder for source and license details.
