"""
lumped.fom: the toolbox FOM in PyBaMM, from exportLumpedModel JSON fixtures
(scripts/make_fom_fixtures.m).
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from lumped.fom import SpecError, load_spec
from lumped.fom.__main__ import main
from lumped.fom.expr import ExpressionError, compile_expression
from lumped.fom.model import Limits, Options
from lumped.fom.run import RunOptions, simulate
from lumped.fom.spec import Param, check_spec

DATA = Path(__file__).parent / "data"
NMC30 = DATA / "cellNMC30.json"
MSMR = DATA / "cellNMCMSMR.json"


def spec_from(d: dict, tmp_path, name="cell.json"):
    f = tmp_path / name
    f.write_text(json.dumps(d))
    return load_spec(f, check=False)


def capacity(spec, e):
    return sum(float(np.squeeze(m["Q"](0, 298.15))) for m in spec[e].materials)


def discharge(spec, rate=1.0, t_on=600.0, t_end=900.0, soc0=80, opts=None):
    """A constant-current pulse then rest; rate in C of the cell-SOC electrode."""
    e = "pos" if spec.half_cell else "neg"
    I = rate * capacity(spec, e)
    t = np.arange(0.0, t_end + 1, 10.0)
    sim = {"time": t, "Iapp": np.where(t < t_on, I, 0.0), "T": 25.0, "SOC0": soc0}
    return simulate(spec, sim, opts or RunOptions(model=Options(mesh=4))), I


@pytest.fixture(scope="module")
def nmc30():
    return json.loads(NMC30.read_text())


# -- spec and expressions ------------------------------------------------------

def test_spec_checks_catch_a_wrong_translation(nmc30):
    spec = load_spec(NMC30)                                 # MATLAB's samples agree
    bad = copy.deepcopy(nmc30["checks"])
    bad[0]["value"] = (np.asarray(bad[0]["value"]) * 1.01).tolist()
    with pytest.raises(SpecError):
        check_spec(spec, bad)


def test_expressions_match_numpy_and_reject_other_names():
    body = "(0.0521.*(1+(T-228))).*x.*(1 + (-1.06).*sqrt(x)).^2 + exp(-x./2) + tanh(x)"
    f = compile_expression(body)
    x, T = 0.7, 300.0
    want = (0.0521 * (1 + (T - 228))) * x * (1 + (-1.06) * np.sqrt(x)) ** 2 + np.exp(-x / 2) + np.tanh(x)
    assert f(x, T) == pytest.approx(want, rel=1e-14)
    for bad in ("sin(x)", "x(1)", "__import__('os')", "x > 1"):
        with pytest.raises(ExpressionError):
            compile_expression(bad)


def test_lut_temperature_slope_and_per_kelvin():
    p = Param("U", "LUT", (np.array([0.0, 1.0]), np.array([1.0, 2.0])), dvalue=np.array([0.1, 0.1]))
    assert p(0.5, 308.15) == pytest.approx(1.5 + 10 * 0.1)
    q = Param("dU", "LUT", (np.array([0.0, 1.0]), np.array([1.0, 2.0])), Tscale=True)
    assert q(2.0, 300.0) == pytest.approx(300.0 * 3.0)       # linear extrapolation, times T


# -- charge conservation across topologies -------------------------------------

def blend_of_halves(d):
    """The negative electrode as two identical materials of half the size each."""
    d = copy.deepcopy(d)
    neg = d["regions"]["neg"]
    m = neg["materials"][0]
    half = copy.deepcopy(m)
    for k, f in (("Q", 0.5), ("Qabs", 0.5), ("QabsCell", 0.5), ("k0", 0.5), ("Cdl", 0.5), ("Rf", 2), ("Rdl", 2)):
        half[k]["value"] = (np.asarray(half[k]["value"]) * f).tolist()
    neg["materials"] = [half, copy.deepcopy(half)]
    z = np.linspace(0, 1, 201)
    soc = Param.from_json("soc", m["soc"], 298.15, 8.3144621)
    U = Param.from_json("Uocp", m["Uocp"], 298.15, 8.3144621)
    neg["params"]["Ueq"] = {"type": "LUT", "value": {"x": z.tolist(), "y": U(soc(z)).tolist()}, "Ea": 0}
    neg["labels"] = ["a", "b"]
    return d


def foil_half_cell(d):
    """The NMC30 positive electrode against a lithium foil."""
    d = copy.deepcopy(d)
    d["regions"]["neg"] = {"kind": "Electrode2D", "labels": [], "materials": [], "params": {
        "k0": {"type": "scalar", "value": 20.0, "Ea": 0}, "alpha": {"type": "scalar", "value": 0.5, "Ea": 0},
        "Rf": {"type": "scalar", "value": 1e-3, "Ea": 0}, "Rdl": {"type": "scalar", "value": 1e-4, "Ea": 0},
        "Cdl": {"type": "scalar", "value": 1.0, "Ea": 0}}}
    return d


@pytest.mark.parametrize("make", ["standard", "msmr", "blend", "foil"])
def test_charge_is_conserved(make, nmc30, tmp_path):
    spec = {"standard": lambda: load_spec(NMC30), "msmr": lambda: load_spec(MSMR),
            "blend": lambda: spec_from(blend_of_halves(nmc30), tmp_path),
            "foil": lambda: spec_from(foil_half_cell(nmc30), tmp_path)}[make]()
    out, I = discharge(spec)
    e = "pos" if spec.half_cell else "neg"
    assert out["termination"] == "final time"
    want = 0.8 - I * 600 / 3600 / capacity(spec, e)
    assert out["SOC"][-1] == pytest.approx(want, abs=2e-4)
    assert np.all(np.isfinite(out["Vcell"]))


def test_blend_of_identical_halves_is_the_single_material(nmc30, tmp_path):
    one, _ = discharge(load_spec(NMC30))
    two, _ = discharge(spec_from(blend_of_halves(nmc30), tmp_path))
    np.testing.assert_allclose(two["Vcell"], one["Vcell"], atol=2e-4)
    assert set(two["neg"]) == {"a", "b"}


def test_flux_form_with_unit_transference_is_the_legacy_form(nmc30, tmp_path):
    d = copy.deepcopy(nmc30)
    d["regions"]["const"]["params"]["t0mNorm"] = {"type": "function", "value": "1 + 0.*x", "Ea": 0}
    legacy, _ = discharge(load_spec(NMC30))
    flux, _ = discharge(spec_from(d, tmp_path))
    np.testing.assert_allclose(flux["Vcell"], legacy["Vcell"], atol=1e-4)
    np.testing.assert_allclose(flux["Thetae"], legacy["Thetae"], atol=1e-4)


# -- charge limits, thermal model and command line -----------------------------

def test_charge_limits_hold_and_soc_stops():
    spec = load_spec(NMC30)
    lim = Limits(phiseMin=0.08, Vmax=3.9, SOCLimit=0.7)
    out, _ = discharge(spec, rate=-2.0, t_on=4000, t_end=4000, soc0=20,
                       opts=RunOptions(model=Options(mesh=4, limits=lim)))
    k1 = np.argmin(np.abs(out["xLocs"]["Phise"] - 1))
    assert out["termination"] == "SOC limit"
    assert out["SOC"][-1] == pytest.approx(0.7, abs=1e-6)
    assert out["Phise"][:, k1].min() >= 0.08 - 1e-6
    assert out["Vcell"].max() <= 3.9 + 1e-6
    assert np.any(out["Iapp"] > out["Ides"] + 1e-3)          # a limit tapered the current


def with_thermal(d):
    d = copy.deepcopy(d)
    d["thermal"] = 1
    for r, (Rth, Cth) in {"neg": (2e-4, 300.0), "sep": (2e-4, 30.0), "pos": (2e-4, 300.0)}.items():
        d["regions"][r]["params"]["Rth"] = {"type": "scalar", "value": Rth, "Ea": 0}
        d["regions"][r]["params"]["Cth"] = {"type": "scalar", "value": Cth, "Ea": 0}
    d["regions"]["const"]["params"]["Rthse"] = {"type": "scalar", "value": 2.0, "Ea": 0}
    return d


def test_thermal_cell_heats_under_load_and_relaxes_to_ambient(nmc30, tmp_path):
    spec = spec_from(with_thermal(nmc30), tmp_path)
    opts = RunOptions(model=Options(mesh=4, thermal=True))
    t = np.arange(0.0, 6001, 20.0)
    I = np.where(t < 1200, capacity(spec, "neg"), 0.0)
    out = simulate(spec, {"time": t, "Iapp": I, "T": 25.0, "SOC0": 90}, opts)
    tau = 2.0 / 2 * 630.0                                    # Rthse in parallel at both ends, total Cth
    assert out["T"][60] > 25.5                                # heated by the 1C pulse
    k = np.searchsorted(t, 1200 + 4 * tau)
    assert abs(out["T"][k] - 25.0) < 0.05 * (out["T"][60] - 25.0)
    assert out["Tcell"].shape[1] == out["xLocs"]["Tcell"].size


def test_command_line_writes_fomout(tmp_path):
    t = np.arange(0.0, 301, 10.0)
    (tmp_path / "in.json").write_text(json.dumps({"time": t.tolist(), "Iapp": [10.0] * t.size, "T": 25, "SOC0": 60}))
    (tmp_path / "o.json").write_text(json.dumps({"mesh": 5, "limits": {"Vmax": 4.3}}))
    code = main(["--model", str(NMC30), "--input", str(tmp_path / "in.json"), "--options", str(tmp_path / "o.json"),
                 "--out", str(tmp_path / "out.mat"), "--progress", str(tmp_path / "p.txt")])
    assert code == 0
    out = loadmat(tmp_path / "out.mat", simplify_cells=True)["out"]
    assert out["Vcell"].size == t.size and out["Phise"].shape[0] == t.size
    assert "100 %" in (tmp_path / "p.txt").read_text().splitlines()[-1]
    code = main(["--model", str(tmp_path / "missing.json"), "--input", str(tmp_path / "in.json"),
                 "--out", str(tmp_path / "err.mat")])
    assert code == 1 and "error" in loadmat(tmp_path / "err.mat")
