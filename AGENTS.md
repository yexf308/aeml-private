# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`:
- `src/numeric/`: neural nets, losses, training loops, sampling, and dataset utilities.
- `src/symbolic/`: manifold geometry and symbolic differential objects.

Experiment entry points are in `experiments/` (for example `ablation_study.py`, `extrapolation_study.py`, `penalty_extrapolation_eval.py`).  
Tests are in `tests/` and mirror module responsibilities (`test_datagen.py`, `test_sampler.py`, etc.).  
Generated artifacts (CSV/JSON/log/plots) are stored at the repo root and in `ablation_results*` directories.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create/activate local environment.
- `pip install --upgrade pip && pip install -r requirements.txt`: install dependencies.
- `python run_experiments.py`: run the default penalty ablation experiment.
- `python -m experiments.ablation_study --surfaces paraboloid --epochs 500`: run configurable ablation study.
- `python -m experiments.extrapolation_study --surface paraboloid --plot`: run extrapolation evaluation and plot output.
- `PYTHONPATH=. pytest -q`: run the full test suite from repo root.
- `PYTHONPATH=. pytest tests/test_datagen.py -q`: run a focused test file.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and PEP 8 naming:
- `snake_case` for functions/variables/modules.
- `PascalCase` for classes/dataclasses (e.g., `TrainingConfig`, `ModelSpec`).

Prefer explicit type hints on public functions and dataclass fields. Keep tensor-shape assumptions clear in variable names and docstrings.

## Testing Guidelines
Use `pytest` with files named `test_*.py` and test functions named `test_*`.  
Add tests next to the behavior you change (numeric vs symbolic). Include deterministic seeds (`seed=...`) for stochastic code and check both shapes and numerical invariants where relevant.

## Commit & Pull Request Guidelines
Follow the repository’s existing commit style: short, imperative subject lines (for example, `Add efficient tangent bundle loss...`).  
Keep commits focused on one logical change. In PRs, include:
- What changed and why.
- Commands used to validate (`PYTHONPATH=. pytest -q`, experiment commands).
- Key metrics/artifacts affected (CSV/plots/logs) and reproducibility parameters (surface, epochs, seed, device).
