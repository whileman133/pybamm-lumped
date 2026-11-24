"""
lumped.fom.spec

Load a compiled BMS3-toolbox cell model exported as JSON (exportLumpedModel.m)
and evaluate its parameter descriptors with NumPy or PyBaMM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pybamm
from scipy.special import expit

from .expr import compile_expression

FORMAT = "bms3-lumped-model"
VERSION = 1


class SpecError(ValueError):
    """A model file this package cannot use."""


@dataclass
class Param:
    """
    One parameter descriptor: value(x, T) * scale * exp(Ea/R*(1/Tref - 1/T)).

    type is 'scalar', 'vector' (one entry per MSMR gallery), 'LUT' (linear
    interpolation with linear extrapolation), 'function' (an expression in x and T)
    or 'MSMR' (a material's OCP or OCP slope from its galleries, exact in T).
    A LUT may add a temperature slope (T - Tref)*dvalue, or be per kelvin (Tscale).
    """
    name: str
    type: str
    value: Any
    Ea: Any = 0.0
    scale: float = 1.0
    xFixed: float | None = None
    dvalue: np.ndarray | None = None
    Tscale: bool = False
    Tref: float = 298.15
    R: float = 8.3144621
    _fn: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, name: str, d: dict, Tref: float, R: float) -> Param:
        value = d["value"]
        if d["type"] == "LUT":
            value = (np.asarray(value["x"], float), np.asarray(value["y"], float))
        elif d["type"] == "vector":
            value = np.atleast_1d(np.asarray(value, float))
        elif d["type"] == "MSMR":
            value = {k: np.atleast_1d(np.asarray(value[k], float)) for k in ("U0", "X", "omega")} \
                | {"quantity": value["quantity"]}
        dvalue = np.asarray(d["dvalue"], float) if d.get("dvalue") is not None else None
        p = cls(name, d["type"], value, d.get("Ea", 0.0), d.get("scale", 1.0) or 1.0, d.get("xFixed"),
                dvalue, bool(d.get("Tscale", False)), Tref, R)
        if p.type == "function":
            p._fn["numpy"] = compile_expression(value, "numpy")
            p._fn["pybamm"] = compile_expression(value, "pybamm")
        elif p.type not in ("scalar", "vector", "LUT", "MSMR"):
            raise SpecError(f"Parameter {name}: unknown descriptor type '{p.type}'.")
        return p

    @property
    def size(self) -> int:
        """Number of entries (galleries) of a vector parameter; 1 otherwise."""
        return self.value.size if self.type == "vector" else 1

    def __call__(self, x=0.5, T=298.15):
        """NumPy value at x and T [K]."""
        x = self.xFixed if self.xFixed is not None else x
        x, T = np.asarray(x, float), np.asarray(T, float)
        if self.type == "scalar":
            base = np.full(np.broadcast(x, T).shape, float(self.value))
        elif self.type == "vector":                               # galleries along the last axis
            T = T[..., None] if T.ndim else T
            base = self.value
        elif self.type == "LUT":
            xs, ys = self.value
            base = _interp_extrap(x, xs, ys)
            if self.dvalue is not None:
                base = base + (T - self.Tref) * _interp_extrap(x, xs, self.dvalue)
            if self.Tscale:
                base = T * base
        elif self.type == "MSMR":
            base = msmr_ocp(x, T, **self.value)
        else:
            base = self._fn["numpy"](x, T) * np.ones(np.broadcast(x, T).shape)
        return self.scale * base * np.exp(np.asarray(self.Ea, float) / self.R * (1 / self.Tref - 1 / T))

    def expr(self, x=0.5, T=None, j: int | None = None):
        """
        PyBaMM expression at x and T (symbols or numbers).

        j picks one entry of a vector parameter; a one-entry vector needs none.
        """
        T = pybamm.Scalar(self.Tref) if T is None else T
        x = self.xFixed if self.xFixed is not None else x
        Ea = self.Ea
        if self.type == "vector":
            if j is None and self.value.size == 1:
                j = 0                                             # one entry: its value, as in MATLAB
            if j is None:
                raise SpecError(f"Parameter {self.name} is a vector; give the entry j.")
            base = pybamm.Scalar(float(self.value[j]))
            Ea = np.atleast_1d(Ea)[j] if np.size(Ea) > 1 else float(np.atleast_1d(Ea)[0])
        elif self.type == "scalar":
            base = pybamm.Scalar(float(self.value))
        elif self.type == "MSMR":
            raise SpecError(f"Parameter {self.name} comes from MSMR galleries; the model solves them directly.")
        elif self.type == "LUT":
            xs, ys = self.value
            xsym = x if isinstance(x, pybamm.Symbol) else pybamm.Scalar(float(x))
            base = pybamm.Interpolant(xs, ys, xsym, interpolator="linear", extrapolate=True)
            if self.dvalue is not None:
                base = base + (T - self.Tref) * pybamm.Interpolant(
                    xs, self.dvalue, xsym, interpolator="linear", extrapolate=True)
            if self.Tscale:
                base = T * base
        else:
            xsym = x if isinstance(x, pybamm.Symbol) else pybamm.Scalar(float(x))
            base = self._fn["pybamm"](xsym, T)
        out = base if self.scale == 1 else self.scale * base
        Ea = float(np.atleast_1d(Ea)[0]) if np.size(Ea) == 1 else Ea
        if np.size(Ea) == 1 and Ea != 0:
            out = out * pybamm.exp(Ea / self.R * (1 / self.Tref - 1 / T))
        return out


def msmr_ocp(theta, T, U0, X, omega, quantity="U", R=8.3144621):
    """
    A material's OCP or OCP slope from its MSMR galleries (MATLAB msmrOCP).

    Parameters
    ----------
    theta : array_like
        Stoichiometries.
    T : array_like
        Temperatures, broadcast against theta. [K]
    U0, X, omega : ndarray of shape (J,)
        Gallery potentials [V], capacity fractions and ideality factors.
    quantity : {"U", "dU"}
        The OCP or its slope dU/dtheta.

    Returns
    -------
    y : ndarray, the broadcast shape of theta and T
        OCP [V] or its slope [V].

    Notes
    -----
    Bisects theta(U) = sum_j X_j/(1 + exp(f*(U - U0_j)/omega_j)), f = F/(R*T).
    Exact for theta in [0.001, 0.999]; beyond, U extends linearly with the end
    slope and dU/dtheta holds its end value.
    """
    theta, TT = np.broadcast_arrays(np.asarray(theta, float), np.asarray(T, float))
    th = np.clip(theta, 1e-3, 1 - 1e-3)[..., None]
    f = 96485.3365 / (R * TT)[..., None]
    span = 60 * omega.max() / f
    lo, hi = U0.min() - span, U0.max() + span
    for _ in range(80):
        mid = (lo + hi) / 2
        above = np.sum(X * expit(-f * (mid - U0) / omega), axis=-1, keepdims=True) > th
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    U = (lo + hi) / 2
    x = X * expit(-f * (U - U0) / omega)
    dU = -1 / (f[..., 0] * np.sum(x / omega * (1 - x / X), axis=-1))
    return U[..., 0] + dU * (theta - th[..., 0]) if quantity == "U" else dU


def _interp_extrap(x, xs, ys):
    """Linear interpolation with linear extrapolation (MATLAB interp1 'linear','extrap')."""
    y = np.interp(x, xs, ys)
    lo, hi = x < xs[0], x > xs[-1]
    y = np.where(lo, ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]), y)
    return np.where(hi, ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]), y)


@dataclass
class Region:
    """
    A cell region: shared parameters and, for an electrode, its materials.

    kinetics lists each material's kinetics, 'MSMR' or 'BV', apart from its OCP;
    empty in a file that predates it (then chosen from k0, as MATLAB does).
    """
    name: str
    kind: str
    params: dict[str, Param]
    materials: list[dict[str, Param]]
    labels: list[str]
    kinetics: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> Param:
        return self.params[name]

    def has(self, name: str) -> bool:
        return name in self.params


@dataclass
class CellSpec:
    """A compiled cell model: flags, regions and constants."""
    name: str
    fingerprint: str
    MSMR: bool
    thermal: bool
    regions: dict[str, Region]
    F: float
    R: float
    Tref: float
    meta: dict

    def __getitem__(self, region: str) -> Region:
        return self.regions[region]

    @property
    def half_cell(self) -> bool:
        """True when the negative electrode is a foil (lithium metal)."""
        return self.regions["neg"].kind == "Electrode2D"


def load_spec(path, check: bool = True) -> CellSpec:
    """
    Read a model file written by exportLumpedModel.m.

    Parameters
    ----------
    path : str or Path
        The JSON model file.
    check : bool
        Compare every parameter with the values MATLAB sampled at export (the
        file's checks); raise SpecError on a mismatch.

    Returns
    -------
    spec : CellSpec
    """
    d = json.loads(Path(path).read_text())
    if d.get("format") != FORMAT:
        raise SpecError(f"{path} is not a {FORMAT} file.")
    if d.get("version", 0) > VERSION:
        raise SpecError(f"{path} has format version {d['version']}; this package reads up to {VERSION}.")
    Tref, R = d["Tref"], d["R"]
    regions = {}
    for name, r in d["regions"].items():
        params = {k: Param.from_json(k, v, Tref, R) for k, v in r.get("params", {}).items()}
        mats = [{k: Param.from_json(k, v, Tref, R) for k, v in m.items()} for m in r.get("materials", [])]
        kin = r.get("kinetics", [])
        kin = [kin] if isinstance(kin, str) else list(kin)
        regions[name] = Region(name, r.get("kind", ""), params, mats, list(r.get("labels", [])), kin)
    spec = CellSpec(d.get("name", ""), d.get("fingerprint", ""), bool(d.get("MSMR", 0)), bool(d.get("thermal", 0)),
                    regions, d["F"], R, Tref, {k: d[k] for k in ("toolbox", "created") if k in d})
    if check:
        check_spec(spec, d.get("checks", []))
    return spec


def check_spec(spec: CellSpec, checks: list[dict], rtol: float = 1e-8) -> None:
    """Raise SpecError when a parameter differs from MATLAB's sampled values."""
    for c in checks:
        reg = spec[c["region"]]
        mat = c.get("material") or None
        params = reg.params if mat is None else reg.materials[mat - 1]
        p = params[c["param"]]
        got = np.atleast_1d(p(np.asarray(c["x"], float), np.asarray(c["T"], float))).ravel()
        want = np.atleast_1d(np.asarray(c["value"], float)).ravel()
        if got.size != want.size:
            got = np.broadcast_to(got, want.shape) if got.size == 1 else got
        if not np.allclose(got, want, rtol=rtol, atol=rtol * np.max(np.abs(want), initial=0.0)):
            where = f"{c['region']}" + (f" material {mat}" if mat else "")
            raise SpecError(
                f"{where} {c['param']}: Python gives {got[:3]} where MATLAB gave {want[:3]} "
                f"(x = {np.atleast_1d(c['x'])[:3]}, T = {np.atleast_1d(c['T'])[:3]})."
            )
