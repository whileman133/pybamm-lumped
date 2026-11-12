"""
lumped.libparams

Build PyBaMM parameter values for LumpedLIBModel from a CellParams object.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pybamm
import scipy.constants as const
from scipy.optimize import brentq
from scipy.special import expit

from .cellparams.model import CellParams, _Container

if TYPE_CHECKING:
    from .libcell import LumpedLIBModel

ELECTRODES = ("neg", "pos")


def lib_model_options(cell: CellParams) -> dict:
    """
    LumpedLIBModel constructor options matching a cell definition:
    {'neg_J': J or None, 'pos_J': J or None}. A cell whose MSMR flag is 1 uses MSMR
    on both electrodes with J = number of galleries in U0; otherwise standard OCP.
    """
    if getattr(cell, "MSMR", 0) != 1:
        return {"neg_J": None, "pos_J": None}
    return {f"{e}_J": np.size(getattr(cell, e).U0.value) for e in ELECTRODES}


def _gallery_values(sec, name: str, J: int, TdegC: float) -> np.ndarray:
    """MSMR per-gallery parameter as a length-J vector (scalars are broadcast)."""
    value = np.atleast_1d(getattr(sec, name)(TdegC=TdegC)).astype(float)
    if value.size == 1:
        value = np.full(J, value[0])
    if value.size != J:
        raise ValueError(f"MSMR parameter '{name}' has {value.size} values; expected {J} galleries.")
    return value


def msmr_thetas(U, U0, X, omega, TdegC: float):
    """
    MSMR stoichiometry theta(U) = sum_j X_j / (1 + exp(f*(U - U0_j)/omega_j)).
    """
    f = const.e * const.N_A / const.R / (TdegC + const.zero_Celsius)  # F/(R*T)
    U = np.asarray(U, dtype=float)[..., None]
    return np.sum(X * expit(-f * (U - U0) / omega), axis=-1)


def msmr_ocp(thetas: float, U0, X, omega, TdegC: float) -> float:
    """
    Invert the MSMR relation: the potential U at which theta(U) = thetas.
    """
    if not 0 < thetas < np.sum(X):
        raise ValueError(f"thetas={thetas} is outside the MSMR range (0, sum(X)={np.sum(X)}).")
    lo, hi = np.min(U0) - 5.0, np.max(U0) + 5.0
    return brentq(lambda U: msmr_thetas(U, U0, X, omega, TdegC) - thetas, lo, hi, xtol=1e-12)


def _msmr_params(sec, J: int, TdegC: float) -> dict:
    return {name: _gallery_values(sec, name, J, TdegC) for name in ("U0", "X", "omega")}


def get_lib_param_values(params: CellParams, model: LumpedLIBModel, TdegC: float = 25.0):
    """
    Construct PyBaMM parameter values for an LIB cell.
    """

    # Collect parameter values.
    values = {
        "kD": params.const.kD(TdegC=TdegC),
        "psi [V]": params.const.psi(TdegC=TdegC),
        "Q [Ah]": params.const.Q(TdegC=TdegC),
        "sep_kappa [Ohm-1]": params.sep.kappa(TdegC=TdegC),
        "sep_qe [Ah]": params.sep.qe(TdegC=TdegC),
    }
    for e in ELECTRODES:
        sec = getattr(params, e)
        values.update({
            f"{e}_sigma [Ohm-1]": sec.sigma(TdegC=TdegC),
            f"{e}_kappa [Ohm-1]": sec.kappa(TdegC=TdegC),
            f"{e}_qe [Ah]": sec.qe(TdegC=TdegC),
            f"{e}_theta0": sec.theta0(TdegC=TdegC),
            f"{e}_theta100": sec.theta100(TdegC=TdegC),
            f"{e}_Rf [Ohm]": sec.Rf(TdegC=TdegC),
            f"{e}_Rdl [Ohm]": sec.Rdl(TdegC=TdegC),
            f"{e}_Cdl [F]": sec.Cdl(TdegC=TdegC),
        })
        J = model.J[e]
        if J is None:
            values.update({
                f"{e}_Ds [s-1]": sec.Ds(TdegC=TdegC),
                f"{e}_k0 [A]": sec.k0(TdegC=TdegC),
                f"{e}_alpha": sec.alpha(TdegC=TdegC),
                f"{e}_Uss [V]": pybamm.Interpolant(
                    sec.Uocp.value.x, sec.Uocp.value.y, getattr(model.var, f"{e}_thetass"), interpolator="linear"
                ),
            })
        else:
            values[f"{e}_Dsref [s-1]"] = sec.Dsref(TdegC=TdegC)
            for name, unit in (("U0", " [V]"), ("X", ""), ("omega", ""), ("k0", " [A]"), ("alpha", "")):
                for j, v in enumerate(_gallery_values(sec, name, J, TdegC)):
                    values[f"{e}_{name}_{j}{unit}"] = v
    return pybamm.ParameterValues(values)


def get_initial(cell, soc0, TdegC, model: LumpedLIBModel | None = None):
    """
    Get PyBaMM initial values for a cell: electrode stoichiometry and OCP at soc0 [%].
    The OCP comes from the MSMR relation for MSMR electrodes, else from the Uocp table.
    MSMR vs. standard is taken from `model` when given, else from the cell definition.
    """
    J = model.J if model is not None else {e: lib_model_options(cell)[f"{e}_J"] for e in ELECTRODES}
    p = _Container()
    for e in ELECTRODES:
        sec = getattr(cell, e)
        theta0, theta100 = sec.theta0(TdegC=TdegC), sec.theta100(TdegC=TdegC)
        thetas = theta0 + (soc0/100)*(theta100 - theta0)
        if J[e] is None:
            Uocp = np.interp(thetas, sec.Uocp.value.x, sec.Uocp.value.y)
        else:
            Uocp = msmr_ocp(thetas, **_msmr_params(sec, J[e], TdegC), TdegC=TdegC)
        setattr(p, f"{e}_thetas", thetas)
        setattr(p, f"{e}_Uocp", Uocp)
    return p
