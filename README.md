# AI for Thermal & Fluid Engineering

## CFD & Thermal Engineering × Engineering Automation × Scientific Machine Learning

**Engineering domains**

- Combustion and in-cylinder flow
- Conjugate heat transfer
- Vehicle aerodynamics
- Battery and electronics cooling

**AI/ML engineering methods**

- Surrogate modeling and optimization
- Explainable machine learning
- Reduced-order modeling
- Physics-informed machine learning
- Digital twins
- Neural operators

## About this work

I'm [Maheswaran Pasupathi](https://github.com/maheswaran-pasupathi), a CFD/thermal simulation engineer with extensive experience in commercial simulation tools such as **STAR-CCM+, CONVERGE and GT-SUITE**, together with hands-on engineering automation using **Python and Java**.

Automation has been a significant part of my engineering work, including workflows around:

- Simulation setup and execution
- Engineering data processing
- Automated post-processing
- Repetitive CAE workflow reduction
- Python/Java-based simulation utilities and workflow automation

I am now extending that **CFD + thermal + automation foundation into AI/ML for engineering simulation problems**.

## The gap I encountered

Learning AI/ML concepts in isolation is only one part of the transition. The more difficult question for me was:

> **How do I translate ML concepts into physically meaningful CFD and thermal engineering workflows?**

Generic ML tutorials explain algorithms well. But a simulation engineer quickly encounters another set of questions:

- How should **velocity, pressure and temperature fields** be represented for machine learning?
- What should become the **features**, and what should become the **engineering target**?
- When is conventional regression sufficient?
- When do **POD/PCA, CNNs, surrogate models, PINNs or neural operators** become appropriate?
- How should an ML prediction be checked against **physics and engineering expectations**, not only R² or RMSE?
- How do we identify whether a model has learned a useful physical relationship or only a statistical correlation?
- Where can AI genuinely complement CFD through **reconstruction, reduced-order modeling, optimization and digital twins**?
- Equally important: **where should conventional physics-based simulation remain the primary engineering tool?**

## How I am approaching it

This repository documents my attempt to answer those questions using **public engineering datasets and reproducible projects**.

The workflow is intentionally physics-first:

1. **Understand the engineering problem**
2. **Establish the physics baseline**
3. **Understand and validate the dataset**
4. **Formulate the ML problem — features, targets and constraints**
5. **Select an appropriate method rather than the most fashionable method**
6. **Evaluate both statistical performance and physical behavior**
7. **Investigate errors, extrapolation and limitations**
8. **Translate the result back into an engineering conclusion**

The objective is not to replace CFD with AI. It is to understand **where data-driven methods can responsibly extend simulation engineering**.

## Why I am sharing the journey

I expect many CFD and thermal engineers face a similar problem:

- The fundamental AI/ML concepts are understandable.
- Python examples and generic datasets are widely available.
- But connecting them to a real CFD, thermal or simulation problem is much less straightforward.

I am therefore building this portfolio in public so that the projects can become:

- **Reproducible engineering examples** rather than isolated ML demonstrations
- A bridge between **CFD/thermal physics and machine learning implementation**
- A place to document what works, what fails, and why
- A starting point for other simulation engineers making a similar transition
- A technical space where the CFD/thermal/Scientific-ML community can question and improve the approaches

This is an **in-progress engineering portfolio**. Projects are updated as they are built and validated, not presented as finished before evidence exists. The priority throughout is **physical interpretation and engineering credibility over model-accuracy claims**.

## Why this repo exists

There is plenty of excellent material for learning ML algorithms and plenty for learning CFD. The gap I want to explore is the engineering layer between them: **turning simulation and experimental data into useful ML problems without losing the physics that makes the result trustworthy.**

Each project follows the same technical logic:

**Engineering problem → Physics baseline → Dataset → ML formulation → Method → Result → Error analysis → Engineering conclusion**

## Portfolio

| # | Project | Physics | AI/ML Method | Status | Preview |
|---|---|---|---|---|---|
| 01 | [In-Cylinder Flow Reconstruction](projects/01-enginebench-piv/) | Engine PIV flow, tumble/vortex structure, cycle variability | POD/PCA → regression → reconstruction | Complete | <img src="projects/01-enginebench-piv/results/stage3_pod_reconstruction.png" width="200"> |
| 02 | [Spray & Combustion ML](projects/02-ecn-spray-combustion/) | Ignition delay, spray penetration, lift-off length | Random Forest / XGBoost + SHAP | Complete | <img src="projects/02-ecn-spray-combustion/results/stage3_shap_summary.png" width="200"> |
| 03 | [Data Center Cooling Surrogate](projects/03-ecoqube-datacenter-cooling/) | Thermal surrogate + cooling control/optimization | Random Forest + grid-search optimization | Complete | <img src="projects/03-ecoqube-datacenter-cooling/results/stage1b_cfd_field_plan.png" width="200"> |
| 04 | [Vehicle Aerodynamics AI](projects/04-drivaernet-aero/) | Drag coefficient prediction from geometry | XGBoost + SHAP explainability | Planned | - |
| 05 | [Electronics Hotspot Prediction](projects/05-electronics-thermal-cnn/) | Power/heat-source layout → temperature map | CNN | Planned | - |
| 06 | [Thermal Digital Twin (flagship)](projects/06-transient-cht-digital-twin/) | Transient conjugate heat transfer, battery/electronics cooling transfer | U-Net → FNO/DeepONet | Planned | - |
| 07 | [Scientific ML / Neural Operators](projects/07-cfdbench-neural-operator/) | Canonical CFD PDE solution fields | Fourier Neural Operator | Planned | - |

## Community

I want this repository to be useful to CFD engineers — experienced or just starting out — who want to develop AI/ML capability **without losing the physics grounding that makes simulation work trustworthy**.

- **Discussions** — questions, alternative approaches, useful datasets and technical debate
- **Issues** — dataset suggestions, method suggestions and identified errors
- [CONTRIBUTING.md](CONTRIBUTING.md) — guidance for contributing a notebook, correction or project improvement

If you're a CFD/thermal engineer exploring ML, or an ML practitioner interested in engineering simulation, you're welcome to challenge the assumptions and results here.

## Let's collaborate

If you are working through a similar transition — or already applying AI/ML successfully to CFD and thermal engineering — I'd like to exchange approaches and lessons learned.

[Connect with me on LinkedIn](https://www.linkedin.com/in/srimahes) to discuss what is working, what is not, and where physics-based simulation and machine learning can complement each other effectively.

## Running this locally

```bash
pip install -r requirements.txt
```

This covers the core dependencies (`numpy`, `h5py`, `matplotlib`, `scikit-learn`) used across projects, plus `kaggle` for datasets accessed through the Kaggle API. Each project README lists its dataset and any project-specific dependencies.

## Datasets used and attribution

All datasets are public/research datasets and remain credited to their original authors. Dataset links, papers and citations are maintained in the individual project READMEs.

**No proprietary or employer data is used in this repository.**

## License

Code in this repository is released under the [MIT License](LICENSE). Datasets remain under their original licenses — see each project folder for source and license details.
