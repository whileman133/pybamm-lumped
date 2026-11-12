"""
python -m lumped.fom: run the toolbox FOM in PyBaMM from files.

Usage
-----
python -m lumped.fom --model cell.json --input sim.json --out out.mat
                     [--options opts.json] [--progress progress.txt]

model    exportLumpedModel.m output.
input    simData as JSON: time [s], Iapp [A] (discharge > 0), T [degC], SOC0 [%].
options  {"mesh": 1-5, "tolerances": "tight"|"medium"|"loose", "maxStep": s,
          "thermal": bool, "limits": {"Vmax": V, "phiseMin": V, ..., "SOCLimit": 0-1}}.
out      MATLAB file with FOMout (simFOM's fields) in variable 'out'; on an
         error, 'error' holds the message and 'report' the traceback.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np
from scipy.io import savemat

from .model import Limits, Options
from .run import RunOptions, simulate
from .spec import load_spec


def run_options(d: dict, progress=None) -> RunOptions:
    """RunOptions from the options JSON."""
    model = Options(thermal=bool(d.get("thermal", False)), limits=Limits.from_dict(d.get("limits")),
                    mesh=int(d.get("mesh", 1)))
    return RunOptions(tolerances=d.get("tolerances", "tight"), max_step=float(d.get("maxStep", 0) or 0),
                      progress=progress, model=model)


def _matlab(v):
    """Values savemat writes as MATLAB types (dicts as structs, arrays as doubles)."""
    if isinstance(v, dict):
        return {k: _matlab(x) for k, x in v.items()}
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (list, tuple, np.ndarray)) and not isinstance(v, str):
        a = np.asarray(v, float)
        return a.reshape(-1, 1) if a.ndim == 1 else a
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lumped.fom", description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--options")
    ap.add_argument("--progress")
    a = ap.parse_args(argv)
    try:
        spec = load_spec(a.model)
        sim = json.loads(open(a.input).read())
        opts = json.loads(open(a.options).read()) if a.options else {}
        out = simulate(spec, sim, run_options(opts, a.progress))
        savemat(a.out, {"out": _matlab(out)}, long_field_names=True, do_compression=True)
        return 0
    except Exception as err:  # reported to MATLAB, which decides how to fail
        savemat(a.out, {"error": f"{type(err).__name__}: {err}", "report": traceback.format_exc()},
                long_field_names=True)
        print(f"{type(err).__name__}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
