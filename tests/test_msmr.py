"""
MSMR electrodes in LumpedLIBModel.
"""
from pathlib import Path

import numpy as np
import pybamm
import pytest

import lumped
from lumped.cellparams.model import Parameter
from lumped.libparams import _msmr_params, get_lib_param_values, msmr_ocp, msmr_thetas

DATA = Path(__file__).parent / "data"

F, R = 96485.33212331001, 8.31446261815324


@pytest.fixture
def msmr_cell():
    cell = lumped.load_cell_params(DATA / "cellNMCMSMR.xlsx")
    # The fixture's neg double layer (Rdl*Cdl = 10 ns) is too stiff for PyBaMM's DAE
    # initialization under a current step; the standard-OCP model fails the same way.
    cell.neg.Cdl = Parameter("Cdl", "F", 1.0, 0.0)
    return cell


def test_lib_model_options():
    msmr = lumped.load_cell_params(DATA / "cellNMCMSMR.xlsx")
    std = lumped.load_cell_params(DATA / "cellNMC30.xlsx")
    assert lumped.lib_model_options(msmr) == {"neg_J": 7, "pos_J": 4}
    assert lumped.lib_model_options(std) == {"neg_J": None, "pos_J": None}


def test_invalid_gallery_count():
    with pytest.raises(ValueError):
        lumped.LumpedLIBModel(pos_J=0)


def test_msmr_has_no_default_parameter_values():
    with pytest.raises(ValueError, match="No default parameter values"):
        lumped.LumpedLIBModel(neg_J=7, pos_J=4).default_parameter_values


@pytest.mark.parametrize("e", ["neg", "pos"])
def test_msmr_ocp_matches_celldef_table(e):
    """The MSMR OCP reproduces the Uocp table stored in the same spreadsheet."""
    cell = lumped.load_cell_params(DATA / "cellNMCMSMR.xlsx")
    sec = getattr(cell, e)
    msmr = _msmr_params(sec, np.size(sec.U0.value), 25)
    x, y = sec.Uocp.value.x, sec.Uocp.value.y
    lo, hi = sorted([sec.theta0(), sec.theta100()])
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    U = np.array([msmr_ocp(t, **msmr, TdegC=25) for t in x[m]])
    np.testing.assert_allclose(U, y[m], atol=1e-4)
    np.testing.assert_allclose(msmr_thetas(U, **msmr, TdegC=25), x[m], atol=1e-9)


def test_single_gallery_msmr_matches_standard_nernst_model():
    """
    One gallery with X=1, omega=1 is exactly the standard model with a Nernst OCP
    U0 + (RT/F) ln((1-theta)/theta), the same single-reaction kinetics, and Ds = Dsref.
    Checks MSMR kinetics, the Us-form of solid diffusion, and its boundary flux.
    """
    T = 298.15
    U0 = {"neg": 0.1, "pos": 3.9}
    cell = lumped.load_cell_params(DATA / "cellNMC30.xlsx")
    Q = cell.const.Q()
    std_values = get_lib_param_values(cell, lumped.LumpedLIBModel(), 25)
    base = {k: v for k, v in std_values.items() if not k.endswith("Uss [V]")}
    base.update({"T [K]": T, "iapp [A]": lambda t: Q * (t < 600)})
    for e in ("neg", "pos"):
        sec = getattr(cell, e)
        th = sec.theta0() + 0.5 * (sec.theta100() - sec.theta0())
        base[f"{e}_thetas0"] = th
        base[f"{e}_Uocp0 [V]"] = U0[e] + R * T / F * np.log((1 - th) / th)

    std = dict(base)
    for e in ("neg", "pos"):
        std[f"{e}_Uss [V]"] = (lambda u0: lambda th: u0 + R * T / F * pybamm.log((1 - th) / th))(U0[e])
    msmr = dict(base)
    for e in ("neg", "pos"):
        msmr.update({
            f"{e}_U0_0 [V]": U0[e], f"{e}_X_0": 1.0, f"{e}_omega_0": 1.0,
            f"{e}_k0_0 [A]": base[f"{e}_k0 [A]"], f"{e}_alpha_0": base[f"{e}_alpha"],
            f"{e}_Dsref [s-1]": base[f"{e}_Ds [s-1]"],
        })

    t_eval = np.arange(0, 1201, 5.0)
    a = pybamm.Simulation(lumped.LumpedLIBModel(), parameter_values=pybamm.ParameterValues(std)).solve(t_eval)
    b = pybamm.Simulation(
        lumped.LumpedLIBModel(neg_J=1, pos_J=1), parameter_values=pybamm.ParameterValues(msmr)
    ).solve(t_eval)
    np.testing.assert_allclose(b["vcell [V]"].entries, a["vcell [V]"].entries, atol=1e-4)
    for k in ("neg_soc [%]", "pos_soc [%]"):
        np.testing.assert_allclose(b[k].entries, a[k].entries, atol=1e-4)


def test_msmr_cell_cc_conserves_charge_and_starts_at_ocv(msmr_cell):
    sim = lumped.sim_cc(msmr_cell, 0.1 * msmr_cell.const.Q(), 60, 58)
    sol = sim.solution
    p0 = lumped.get_initial(msmr_cell, 60, 25)
    assert sol["vcell [V]"].entries[0] == pytest.approx(p0.pos_Uocp - p0.neg_Uocp, abs=0.05)
    for e in ("neg", "pos"):
        assert sol[f"{e}_soc [%]"].entries[-1] == pytest.approx(58, abs=0.02)


def test_msmr_cell_eis(msmr_cell):
    freq = np.logspace(-2, 3, 6)
    Z = np.array(lumped.sim_eis(msmr_cell, freq, 50).solution)
    assert np.all(np.isfinite(Z))
    assert np.all(Z.real > 0) and np.all(Z.imag < 0)
