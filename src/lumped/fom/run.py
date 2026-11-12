"""
lumped.fom.run

Simulate a toolbox input profile (simFOM's simData) with FOMModel and return
simFOM's output fields (FOMout).
"""
from __future__ import annotations

import time as _time
from importlib.metadata import version
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybamm

from .model import FOMModel, Limits, Options
from .spec import CellSpec

TOLERANCES = {"tight": (1e-6, 1e-8), "medium": (1e-5, 1e-7), "loose": (1e-4, 1e-6)}
TSHIFT = 0.0025   # outputs are sampled this long after each sample time, as in simFOM


@dataclass
class SimData:
    """
    Input profile, as simFOM's simData.

    time and Iapp [A] are a zero-order hold (the last current is unused);
    T [degC] is a scalar or a zero-order hold on time; SOC0 is in percent.
    """
    time: np.ndarray
    Iapp: np.ndarray
    T: np.ndarray
    SOC0: float

    @classmethod
    def from_dict(cls, d: dict) -> SimData:
        t = np.atleast_1d(np.asarray(d["time"], float)).ravel()
        i = np.atleast_1d(np.asarray(d["Iapp"], float)).ravel()
        T = np.atleast_1d(np.asarray(d.get("T", 25.0), float)).ravel()
        if t.size != i.size:
            raise ValueError("simData.time and simData.Iapp differ in length.")
        if T.size not in (1, t.size):
            raise ValueError("simData.T is a scalar or one value per time sample.")
        if t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("simData.time needs two or more increasing samples.")
        return cls(t, i, T if T.size > 1 else np.full(t.size, T[0]), float(np.squeeze(d["SOC0"])))


@dataclass
class RunOptions:
    """Solver options: tolerances preset, maximum step [s] (0: the solver's choice), progress file."""
    tolerances: str = "tight"
    max_step: float = 0.0
    progress: str | Path | None = None
    model: Options = field(default_factory=Options)


class _Progress:
    """Writes 'NN %' lines to a progress file (the job queue reads the last line)."""

    def __init__(self, path, t0, t1):
        self.path, self.t0, self.t1, self.last = path, t0, t1, -1.0

    def __call__(self, t, final=False):
        if not self.path:
            return
        pct = 100.0 * (t - self.t0) / max(self.t1 - self.t0, 1e-12)
        if final or pct - self.last >= 1:
            self.last = pct
            with open(self.path, "a") as f:
                f.write(f"PyBaMM (IDAKLU): {min(pct, 100):.0f} %\n")


CHUNK = 100       # most samples per solver call, so a failure keeps the results so far


def _segments(sd: SimData):
    """(t_start, t_end, Ides, Tamb [K], sample indices) for runs of equal current and temperature."""
    segs, k, n = [], 0, sd.time.size - 1
    while k < n:
        j = k + 1
        while j < n and j - k < CHUNK and sd.Iapp[j] == sd.Iapp[k] and sd.T[j] == sd.T[k]:
            j += 1
        segs.append((sd.time[k], sd.time[j], sd.Iapp[k], sd.T[k] + 273.15, np.arange(k, j)))
        k = j
    return segs


def simulate(spec: CellSpec, sim: SimData | dict, opts: RunOptions | None = None) -> dict:
    """
    Run an input profile and return simFOM-style outputs.

    Parameters
    ----------
    spec : CellSpec
        Compiled cell model.
    sim : SimData or dict
        Input profile (time [s], Iapp [A], T [degC], SOC0 [%]).
    opts : RunOptions
        Solver, mesh, thermal and charge-limit options.

    Returns
    -------
    out : dict
        FOMout fields: time, Iapp, Ides, Vcell, T, SOC, negSOC, posSOC, Phise,
        Phie, Thetae, Ifdl, negIf, posIf, neg/pos per-material fields, xLocs,
        Tcell (thermal) and status fields (termination, message, wallTime).
    """
    sd = sim if isinstance(sim, SimData) else SimData.from_dict(sim)
    opts = opts or RunOptions()
    wall = _time.perf_counter()
    model = FOMModel(spec, sd.SOC0 / 100, sd.T[0] + 273.15, opts.model)
    mesh = pybamm.Mesh(model.default_geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    outvars = _register_outputs(model)
    disc.process_model(model)
    rtol, atol = TOLERANCES[opts.tolerances]
    solver_opts = {"max_step_size": float(opts.max_step)} if opts.max_step > 0 else {}
    solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol, options=solver_opts)
    limits = opts.model.limits
    modes = [n for n, _ in limits.bounds]
    mode = None
    progress = _Progress(opts.progress, sd.time[0], sd.time[-1])

    rows: dict[str, list] = {k: [] for k in outvars}
    times: list[float] = []
    state = None
    termination, message = "final time", ""
    stop = False
    prev_Ides = prev_Iapp = 0.0
    for (ta, tb, Ides, Tamb, idx) in _segments(sd):
        tq = sd.time[idx]
        t = ta
        if mode is not None and (np.sign(Ides) != np.sign(prev_Ides) or Ides > prev_Iapp):
            mode = None                                      # the new demand no longer pushes the limit
        while t < tb - 1e-12:
            inputs = {"Ides [A]": Ides, "Tamb [K]": Tamb}
            inputs.update({f"mode {n}": float(n == mode) for n in modes})
            ts_want = np.minimum(tq[tq + TSHIFT > t + 1e-12] + TSHIFT, tb)
            t_interp = np.unique(np.concatenate([[0.0], ts_want - t, [tb - t]]))
            try:
                sol = solver.step(state, model, tb - t, inputs=inputs, t_interp=t_interp, save=False)
            except pybamm.SolverError as err:
                termination, stop = "failed", True
                message = f"PyBaMM stopped between t = {t:.6g} s and {tb:.6g} s: {err}"
                break
            ts = sol.t
            want = np.concatenate([np.minimum(tq + TSHIFT, tb), [sd.time[-1]] if tb == sd.time[-1] else []])
            keep = np.isin(np.round(ts, 9), np.round(want, 9))
            keep &= ts > (times[-1] + TSHIFT + 1e-9 if times else -np.inf)
            for k, v in outvars.items():
                vals = sol[k].entries
                vals = vals.T if vals.ndim == 2 else vals
                rows[k].extend(np.atleast_2d(vals[keep]) if vals.ndim == 2 else vals[keep])
            times.extend(np.where(ts[keep] >= sd.time[-1] - 1e-12, ts[keep], ts[keep] - TSHIFT))
            progress(ts[-1])
            t = ts[-1]
            state = _carrier(sol, model, inputs)
            term = sol.termination or ""
            if term.startswith("event:"):
                name = term.split("event:", 1)[1].strip()
                if name == "SOC limit":
                    termination, stop = "SOC limit", True
                    _append_final(sol, outvars, rows, times)
                    break
                if name.startswith("limit release"):
                    mode = None
                elif name.startswith("limit "):
                    mode = name.split("limit ", 1)[1]
                    if mode.startswith("thetae"):
                        state = _reset_integrator(state, model)
            prev_Ides = Ides
            prev_Iapp = float(sol["Iapp [A]"].entries[-1])
        if stop:
            break
    progress(times[-1] if times else sd.time[0], final=True)
    if not times:
        raise RuntimeError(message or "PyBaMM produced no output.")
    out = _fomout(model, rows, times, outvars)
    out.update(termination=termination, message=message, wallTime=_time.perf_counter() - wall,
               engine=f"pybamm {pybamm.__version__}, pybamm-lumped {version('pybamm-lumped')}")
    return out


def _carrier(sol, model, inputs):
    """A one-point solution at the end of sol, so step() continues from it after an event."""
    y = sol.all_ys[-1][:, -1:]
    s = pybamm.Solution(np.array([sol.t[-1]]), y, model, inputs, termination="final time")
    return s


def _reset_integrator(state, model):
    """Zero the theta_e PI integrator (bumpless transfer on entering a theta_e limit)."""
    sl = model.y_slices[model.piXi][0]
    y = state.all_ys[-1].copy()
    y[sl] = 0
    return pybamm.Solution(state.t, y, model, state.all_inputs[-1], termination="final time")


def _append_final(sol, outvars, rows, times):
    """Add the event time as the last output sample."""
    if times and abs(times[-1] - sol.t[-1]) < 1e-9:
        return
    for k in outvars:
        vals = sol[k].entries
        vals = vals.T if vals.ndim == 2 else vals
        rows[k].append(vals[-1])
    times.append(sol.t[-1])


def _register_outputs(model: FOMModel) -> dict:
    """Name the variables the outputs need (before discretisation)."""
    v = {}
    v["Iapp [A]"] = model.Iapp if isinstance(model.Iapp, pybamm.Variable) else model.Ides
    v["Ides [A]"] = model.Ides
    v["Vcell [V]"] = model.Vcell
    v["SOC"] = model.SOC
    v["Tamb [K]"] = model.Tamb
    for e in model.electrodes:
        v[f"{e} thetas avg"] = model.thavg[e]
        v[f"{e} phise"] = model.out[f"{e}_phise"]
        v[f"{e} phise edge"] = pybamm.boundary_value(model.out[f"{e}_phise"], "right" if e == "neg" else "left")
        v[f"{e} ifdl"] = model.ifdl_e[e]
        v[f"{e} phis"] = model.phis[e]
        v[f"{e} if"] = sum(m["i_f"] for m in model.mat[e])
        for i, m in enumerate(model.mat[e]):
            for q in ("thetass", "i_f", "idl", "eta", "ifdl"):
                v[f"{e}{i + 1} {q}"] = m[q] + 0 * model.phis[e]       # on the electrode domain
    v["thetae"] = model.thetae
    v["thetae left"] = pybamm.boundary_value(model.thetae, "left")
    v["thetae right"] = pybamm.boundary_value(model.thetae, "right")
    v["phie"] = model.phie
    if model.half:
        v["foil phise"] = model.out["foil_phise"]
        v["foil ifdl"] = model.foil["ifdl"]
    if model.options.thermal:
        v["Tcell"] = model.Tc
        v["Tcell left"] = pybamm.boundary_value(model.Tc, "left")
        v["Tcell right"] = pybamm.boundary_value(model.Tc, "right")
    model.variables.update(v)
    return {k: None for k in v}


def _fomout(model: FOMModel, rows, times, outvars) -> dict:
    """simFOM's FOMout fields from the collected rows ([Nt x Nx] matrices)."""
    mesh_x = {r: model.xlim[r] for r in model.regions}
    nodes = {}
    # Node positions: uniform cells in each region.
    counts = {r: model.default_var_pts[model.x[r]] for r in model.regions}
    for r in model.regions:
        a, b = mesh_x[r]
        h = (b - a) / counts[r]
        nodes[r] = a + h * (np.arange(counts[r]) + 0.5)
    def A(k):
        rs = [np.atleast_1d(np.asarray(r, float)) for r in rows[k]]
        return np.vstack(rs) if rs and rs[0].size > 1 else np.asarray(rows[k], float)
    t = np.asarray(times, float)
    out = {"time": t, "Iapp": A("Iapp [A]"), "Ides": A("Ides [A]"), "Vcell": A("Vcell [V]"),
           "SOC": A("SOC"), "foilNeg": bool(model.half)}
    xl = {}
    # Electrolyte: boundary values at the current collectors plus cell centres.
    xe = np.concatenate([[mesh_x[model.regions[0]][0]], *[nodes[r] for r in model.regions],
                         [mesh_x[model.regions[-1]][1]]])
    out["Thetae"] = np.column_stack([A("thetae left"), A("thetae"), A("thetae right")])
    xl["Thetae"] = xe
    xphie = np.concatenate([nodes[r] for r in model.regions])
    out["Phie"] = A("phie")
    xl["Phie"] = xphie
    # Electrode fields: neg nodes then its sep edge; pos sep edge then its nodes.
    ph, xph, ifs, xif, ifdl = [], [], [], [], []
    for e in model.electrodes:
        P = A(f"{e} phise")
        E = A(f"{e} phise edge")[:, None]
        edge = mesh_x[e][1] if e == "neg" else mesh_x[e][0]
        if e == "neg":
            ph.append(np.hstack([P, E])); xph.append(np.concatenate([nodes[e], [edge]]))
        else:
            ph.append(np.hstack([E, P])); xph.append(np.concatenate([[edge], nodes[e]]))
        out[f"{e}If"] = A(f"{e} if")
        out[f"{e}Ifdl"] = A(f"{e} ifdl")
        out[f"{e}Phise"] = P
        out[f"{e}Phis"] = A(f"{e} phis")
        out[f"{e}SOC"] = A(f"{e} thetas avg")
        ifs.append(out[f"{e}If"]); xif.append(nodes[e]); ifdl.append(out[f"{e}Ifdl"])
        labels = model.spec[e].labels or [f"mat{i + 1}" for i in range(len(model.mat[e]))]
        sub = {}
        for i, _ in enumerate(model.mat[e]):
            lab = _field(labels[i] if i < len(labels) else f"mat{i + 1}")
            sub[lab] = {"Thetass": A(f"{e}{i + 1} thetass"), "If": A(f"{e}{i + 1} i_f"),
                        "Idl": A(f"{e}{i + 1} idl"), "Eta": A(f"{e}{i + 1} eta"),
                        "Ifdl": A(f"{e}{i + 1} ifdl"), "xLoc": nodes[e]}
        out[e] = sub
        out[f"{e}Thetass"] = sub[next(iter(sub))]["Thetass"]
    if model.half:
        ph.insert(0, A("foil phise")[:, None]); xph.insert(0, np.array([0.0]))
        out["neg"] = {}
    out["Phise"] = np.hstack(ph)
    xl["Phise"] = np.concatenate(xph)
    out["If"] = np.hstack(ifs)
    out["Ifdl"] = np.hstack(ifdl)
    xl["If"] = np.concatenate(xif)
    xl["Ifdl"] = xl["If"]
    if model.options.thermal:
        Tc = np.column_stack([A("Tcell left"), A("Tcell"), A("Tcell right")]) - 273.15
        out["Tcell"] = Tc
        xl["Tcell"] = xe
        out["T"] = Tc[:, -1]
    else:
        out["T"] = A("Tamb [K]") - 273.15
    out["xLocs"] = xl
    return out


def _field(label: str) -> str:
    """A MATLAB struct field name from a material label."""
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in str(label))
    return s if s and s[0].isalpha() else f"m_{s}"
