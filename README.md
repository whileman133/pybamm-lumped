# PyBaMM-lumped

PyBaMM models for Plett and Trimboli's BMS III cell models, the lumped-parameter
lithium-ion and lithium-metal cell models. The package provides:

- `LumpedLIBModel(neg_J=None, pos_J=None)`: lithium-ion cell (neg / sep / pos), Butler–Volmer kinetics.
  Each electrode uses standard OCP with constant Ds (`J=None`, default) or MSMR with `J` galleries (see below).
- `LumpedLMBModel(J)`: lithium-metal cell (eff / pos), MSMR OCP and kinetics with `J` galleries, stoichiometry-dependent Ds.
- A loader for cell-definition workbooks (`.xlsx`), simulation helpers (CC, GITT, arbitrary
  profile, drive cycle, EIS), and COMSOL `.mat` loaders.

Cell workbooks are not part of this package; they come from the BMS3 toolbox.

## Install

### 1. Install uv

[uv](https://docs.astral.sh/uv/) manages Python, the virtual environment, and dependencies.
You do not need to install Python yourself; uv downloads Python 3.12 when needed.

**macOS** (Terminal):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or, with Homebrew: `brew install uv`.

**Windows** (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

or, with WinGet: `winget install --id=astral-sh.uv -e`.

Then close and reopen the terminal so `uv` is on your `PATH`, and check that it works:

```bash
uv --version
```

### 2. Set up the project

From inside this folder (Terminal on macOS, PowerShell on Windows):

```bash
cd pybamm-lumped
uv sync
```

That's all. `uv sync` downloads Python 3.12 if needed, creates a `.venv` folder, and installs the
exact dependency versions pinned in `uv.lock` plus this package (in editable mode, so changes
under `src/lumped/` take effect immediately). Run it again whenever `uv.lock` changes.

### 3. Run scripts

Put your scripts anywhere inside this folder and run them with `uv run`, from this folder or any
subfolder:

```bash
uv run examples/run_cc.py
uv run my-scripts/my_analysis.py
uv run pytest                     # the test suite
```

`uv run` always uses this project's environment, so there is nothing to activate. For an interactive
session, use `uv run python`, or `uv run --with jupyter jupyter lab` for notebooks.

**Sharing this folder:** run

```bash
uv run scripts/make-dist-zip.py
```

to write `dist/<date>-pybamm-lumped-<version>.zip` (about 1 MB). It leaves out `.venv/` (large and
specific to one computer and operating system), `.git/`, caches, and OS files. Recipients unzip it and
follow steps 1–3 above; everyone creates their own `.venv` with `uv sync`.

### Using it from another uv project (optional)

```bash
uv add /path/to/pybamm-lumped
```

## Usage

```python
import pybamm
import lumped

# Simulations driven by a cell-definition workbook (LIB model).
cell = lumped.load_cell_params("cellNMC30.xlsx")
Q = cell.const.Q()
sim = lumped.sim_cc(cell, 1*Q, soc0=100, socf=80, t_rest=600)
sim.plot()

eis = lumped.sim_eis(cell, freq=[0.01, 0.1, 1, 10, 100], soc0=50)
eis.nyquist_plot()

# Models with built-in default parameter values.
pybamm.Simulation(lumped.LumpedLIBModel()).solve([0, 600])
pybamm.Simulation(lumped.LumpedLMBModel(J=2)).solve([0, 600])
```

`LumpedLIBModel`'s defaults are the `cellNMC30` test fixture at 25 °C and 50 % SOC, stored in
`src/lumped/lib-defaults.json`; `uv run scripts/make-lib-defaults.py` regenerates that file.

## MSMR in the LIB model

`LumpedLIBModel(neg_J=7, pos_J=4)` uses MSMR on both electrodes (either may be `None` for standard OCP).
For an MSMR electrode, the model:

- solves solid diffusion for the particle potential `Us` (Baker–Verbrugge diffusivity,
  `Ds = Dsref·f·θ(1−θ)·(−dU/dθ)`), with `θ = Σ X_j / (1 + exp(f(U − U0_j)/ω_j))`;
- sums per-gallery Butler–Volmer currents with the BMS3-toolbox (`genFOM`) exchange current
  `i0_j = k0_j · xss_j^ω_j · exp((1−α_j) f (Uss − U0_j)) · θe^(1−α_j)`;
- takes parameters `{e}_U0_j [V]`, `{e}_X_j`, `{e}_omega_j`, `{e}_k0_j [A]`, `{e}_alpha_j`, `{e}_Dsref [s-1]`
  (in place of `{e}_Ds`, `{e}_k0`, `{e}_alpha`, `{e}_Uss`, `{e}_thetas0`), and adds output `{e}_Us [V]`.

`LumpedLMBModel` uses the same unscaled `k0_j`. To convert a `k0` scaled by `(X_j/2)^ω_j`
(the convention in Hileman's thesis and in earlier copies of `lmbcell.py`):
`k0_j = k0_j_scaled / (X_j/2)**ω_j`.

The `sim_*` helpers pick the model variant from the spreadsheet (`MSMR = 1` in the `general` section;
see `lumped.lib_model_options(cell)`), and `get_initial` inverts the MSMR relation for the initial OCP.
MSMR electrodes have no built-in default parameter values.

Caveat for the test fixture `tests/data/cellNMCMSMR.xlsx` (from the BMS3 toolbox): its `neg Cdl = 1e-5 F`
(`Rdl·Cdl = 10 ns`) is too stiff for PyBaMM's solver under a current step (the standard model fails
the same way), and its positive-electrode `k0` values give an exchange current of ~1e-5 A at 50% SOC,
i.e. ~0.5 V of overpotential at C/10. The tests use it with `neg Cdl = 1 F`.

## Module map (vs. the original scripts)

| Original | Package |
|---|---|
| `lumped/base.py`, `libcell.py`, `lmbcell.py` | `lumped.base`, `lumped.libcell`, `lumped.lmbcell` (same import paths) |
| `cellparams/` | `lumped.cellparams` |
| `util.get_lib_param_values`, `util.get_initial` | `lumped.libparams` |
| `util.sim_*` | `lumped.sim` (also exported from `lumped`) |
| `util.FOMOutput`, `util.load_mat_comsol_*` | `lumped.comsol` |
| `XLSX_CELLDEFS/*.xlsx` | Not bundled; `cellNMC30` and `cellNMCMSMR` are test fixtures in `tests/data/` |
| `pybammeis/` | `lumped._vendor.pybammeis` (modified; see its `VENDORED.md`) |

Behavior changes from the original scripts:

- `sim_drive(cell, cycle_csv, ...)` takes a path to a two-column CSV (time [s], current [A]).
  The original read `CSV_DRIVECYCLES/<name>.csv` and referenced a non-existent `cell.cst.QAh`,
  so it could not run.
- `load_cell_params` also accepts MATLAB-style vectors (`[0.1  0.2 0.3]`), which
  `cellNMCMSMR.xlsx` uses.
- Nothing depends on the working directory any more.
- `LumpedLMBModel` MSMR `k0_j` is unscaled (no `(X_j/2)^ω_j` factor), matching the LIB model and the
  BMS3 toolbox. Its default `pos_k0_j` values were converted, so default simulations are unchanged;
  parameter sets fitted with the scaled form must be converted (see MSMR section).

## Tests

`tests/test_regression.py` checks that CC, GITT, EIS, and the default-parameter runs of both models
match outputs from the original code (`tests/data/reference-outputs.npz`) to a relative tolerance of 1e-6.
If you intentionally change model behavior, regenerate that file and say why in the commit.

`pybamm` is pinned to `~=25.1.1` because the models and the vendored EIS code use PyBaMM internals.

## License

MIT; see `LICENSE`. The vendored `pybammeis` code keeps its own MIT license
(`src/lumped/_vendor/pybammeis/LICENSE.md`).
