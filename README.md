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

## The toolbox FOM (`lumped.fom`)

`lumped.fom` solves the BMS3 toolbox's lumped full-order model, the equations
`genFOM.m` builds in COMSOL, from a compiled cell model that the toolbox exports
as JSON (`exportLumpedModel.m`). MATLAB users call it through `simPyBaMM.m`, which
installs this package by itself (`setupPyBaMM.m`); this section covers the Python
side.

```python
from lumped.fom import load_spec
from lumped.fom.model import Limits, Options
from lumped.fom.run import RunOptions, simulate

spec = load_spec("cell.json")                    # checks every parameter against MATLAB
sim = {"time": t, "Iapp": I, "T": 25.0, "SOC0": 20}   # simFOM's simData: ZOH, discharge > 0
out = simulate(spec, sim, RunOptions(model=Options(mesh=1, thermal=False,
                                                    limits=Limits(Vmax=4.2, phiseMin=0.0))))
out["Vcell"], out["Phise"], out["xLocs"]["Phise"]  # simFOM's FOMout fields
```

`Options(mesh=k)` picks a preset, and `Options(elements={"particle": 300})` overrides any of
its counts (`neg`, `sep`, `pos`, `particle`). The presets give 20, 15, 10, 8 and 5 elements
per region and 400, 200, 100, 50 and 30 per particle radius (levels 1 to 5). Particles get
many elements because a diffusivity that changes by orders of magnitude over a narrow range of
θ makes the surface stoichiometry step between elements, and the current then ripples along x.
O'Regan's graphite gives a voltage ripple of 0.52 mV at 50 particle elements, 0.029 mV at 200
and 0.012 mV at 400.

From a shell, `python -m lumped.fom --model cell.json --input sim.json --out out.mat`
writes FOMout to a MATLAB file (`--options` and `--progress` are optional).

What the model covers:

- Standard and MSMR electrodes, blended electrodes (several materials sharing
  φs, φe and θe), and a lithium-foil negative electrode (half cells).
- Every parameter as a scalar, an MSMR gallery vector, a lookup table (linear,
  with linear extrapolation, a temperature slope or a per-kelvin scale), or an
  expression in x and T, each times its scale and Arrhenius factor. Each takes the
  argument genFOM gives it (θe, the surface stoichiometry, or 0.5).
- Explicit or Baker–Verbrugge solid diffusivity, the series Rdl–Cdl double layer
  behind Rf, and the flux form of electrolyte transport when the cell gives
  `t0mNorm`.
- The lumped thermal model (`addThermalModel.m`) and the charge limits
  (`addChargeLimits.m`): Vmax, φse at the neg/sep and sep/pos boundaries, θe at the
  current collectors (a PI controller on the current), and an SOC stop.

Validation and differences from COMSOL:

- On the SKON603 cell (a two-material negative electrode), a 2C fast charge
  matches the stored COMSOL run to 1.5 mV RMS in voltage and 0.06 mV RMS in φse
  at the separator. It runs in 20 s at the finest mesh, against COMSOL's 125 s.
- The tests check charge conservation for each topology, that a blend of two
  identical half-size materials equals the single material, and that the flux form
  with unit transference equals the legacy form.
- Finite volumes replace COMSOL's finite elements, and particles use a uniform
  radial mesh; the surface-flux condition then delivers exactly the faradaic
  charge, however steeply Ds varies.
- Outputs are sampled 2.5 ms after each sample time, as simFOM does by default.
- A charge limit takes over at the instant it binds, found by a solver event,
  rather than at COMSOL's once-a-second check. COMSOL's exchange-current
  relaxation (`relaxExchangeCurrentSingularity`) has no counterpart.

`scripts/make_fom_fixtures.m` writes the test fixtures (`tests/data/*.json`) from a
toolbox checkout.

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
Its IDAKLU solver needs the matching `pybammsolvers` (0.1.0) and the CasADi release that
solver was built with (3.6.7), so both are pinned too.

## License

MIT; see `LICENSE`. The vendored `pybammeis` code keeps its own MIT license
(`src/lumped/_vendor/pybammeis/LICENSE.md`).
