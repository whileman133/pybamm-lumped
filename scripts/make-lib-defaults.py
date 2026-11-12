"""
make-lib-defaults.py

Write LumpedLIBModel's default parameter values (src/lumped/lib-defaults.json)
from the cellNMC30 test fixture, evaluated at 25 degC with the cell at 50 % SOC.

Usage: uv run scripts/make-lib-defaults.py
"""
import json
from numbers import Number
from pathlib import Path

import pybamm

import lumped
from lumped.libparams import get_initial, get_lib_param_values

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "data" / "cellNMC30.xlsx"
OUT = ROOT / "src" / "lumped" / "lib-defaults.json"
TDEGC, SOC0 = 25.0, 50.0


def build() -> dict:
    """Default values: scalars, OCP tables, and the initial state."""
    cell = lumped.load_cell_params(FIXTURE)
    model = lumped.LumpedLIBModel()
    builtin = set(pybamm.ParameterValues({}).keys())
    values = get_lib_param_values(cell, model, TdegC=TDEGC)
    scalars = {k: float(v) for k, v in values.items() if k not in builtin and isinstance(v, Number)}
    init = get_initial(cell, SOC0, TDEGC, model)
    for e in ("neg", "pos"):
        scalars[f"{e}_thetas0"] = float(getattr(init, f"{e}_thetas"))
        scalars[f"{e}_Uocp0 [V]"] = float(getattr(init, f"{e}_Uocp"))
    scalars["T [K]"] = TDEGC + 273.15
    tables = {e: {"x": getattr(cell, e).Uocp.value.x.tolist(), "y": getattr(cell, e).Uocp.value.y.tolist()}
              for e in ("neg", "pos")}
    return {"source": "cellNMC30", "TdegC": TDEGC, "soc0": SOC0, "scalars": scalars, "Uocp": tables}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")
