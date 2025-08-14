# AHP4WtE_Model
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16875959.svg)](https://doi.org/10.5281/zenodo.16875959)

Python implementation of an Analytic Hierarchy Process model for selecting Waste-to-Energy technologies.

AHP4WTE is an open, reproducible Python implementation of the Analytic Hierarchy Process (AHP) for evaluating and ranking Waste‑to‑Energy (WtE) technologies. It provides:
- a clear hierarchical model (Goal → Criteria → Subcriteria → Indicators → Alternatives),
- a solver based on Saaty’s principal eigenvector method,
- automatic consistency checks (λ_max, CI, CR),
- the ability to transform **raw quantitative data** (e.g., CAPEX) directly into pairwise comparisons,
- a worked example notebook to get started quickly.

## Features
- Build a customizable AHP hierarchy for WtE (or any MCDM problem).
- Accept both **quantitative** inputs (e.g., EUR/t, kg CO₂/t) and **qualitative** pairwise judgments.
- Convert cost‑type metrics to pairwise comparisons (lower‑is‑better), then compute priority vectors.
- Check matrix consistency with standard AHP statistics.
- Export pairwise matrices, weights, and rankings to CSV; generate plots.

## Repository Structure
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── CHANGELOG.md
│
├── notebooks/
│ └── ahp_capex_example.ipynb # Walk‑through example (CAPEX)
│
├── src/
│ ├── data_handler.py # Load & validate CSV inputs
│ └── ahp_solver.py # Eigenvector + CI/CR utilities (optional)
│
├── data/
│ └── capex_input.csv # Example raw input for 11 technologies
│
├── results/ # (Optional) saved weights/plots
└── figures/ # (Optional) figures for paper

##  Quick Start

### 1) Install
```bash
# Python 3.8+ recommended
pip install -r requirements.txt

## How to cite

If you use AHP4WTE in your research, please cite:

Patil, S. (2025). *AHP4WTE: Python AHP model for Waste-to-Energy technology selection* (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.16875959

**BibTeX**
```bibtex
@software{AHP4WTE_zenodo_16875959,
  author       = {Patil, Shivaraj Chandrakant},
  title        = {AHP4WTE: Python AHP model for Waste-to-Energy technology selection},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {1.0.1},
  doi          = {10.5281/zenodo.16875959},
  url          = {https://doi.org/10.5281/zenodo.16875959}
}
