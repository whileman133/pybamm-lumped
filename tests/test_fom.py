"""
lumped.fom: the toolbox FOM in PyBaMM, from exportLumpedModel JSON fixtures
(scripts/make_fom_fixtures.m).
"""
import copy
import json
from pathlib import Path

import numpy as np
import pybamm
import pytest
from scipy.io import loadmat

from lumped.fom import SpecError, load_spec
from lumped.fom.__main__ import main
from lumped.fom.expr import ExpressionError, compile_expression
from lumped.fom.model import FOMModel, Limits, Options
from lumped.fom.run import CHUNK, RunOptions, _step, simulate
from lumped.fom.spec import Param, check_spec

DATA = Path(__file__).parent / "data"
NMC30 = DATA / "cellNMC30.json"
MSMR = DATA / "cellNMCMSMR.json"
SKON = DATA / "cellSKON603.json"


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




def test_a_one_entry_vector_reads_as_its_value():
    p = Param("Rf", "vector", np.array([2e-3]))
    assert float(p.expr(0.5).evaluate()) == pytest.approx(2e-3)
    with pytest.raises(SpecError):
        Param("k0", "vector", np.array([1.0, 2.0])).expr(0.5)

def test_msmr_ocp_is_exact_at_each_temperature():
    # One gallery: U = U0 + (omega/f)*ln(X/theta - 1), dU/dtheta = -(omega/f)*X/(theta*(X - theta)).
    g = {"U0": [0.2], "X": [0.9], "omega": [1.5]}
    th, T = np.array([0.1, 0.45, 0.8]), 263.15
    w_f = 1.5 * 8.3144621 * T / 96485.3365
    U = Param.from_json("Uocp", {"type": "MSMR", "value": g | {"quantity": "U"}, "Ea": 0}, 298.15, 8.3144621)
    dU = Param.from_json("dUocp", {"type": "MSMR", "value": g | {"quantity": "dU"}, "Ea": 0}, 298.15, 8.3144621)
    assert U(th, T) == pytest.approx(0.2 + w_f * np.log(0.9 / th - 1), abs=1e-9)
    assert dU(th, T) == pytest.approx(-w_f * 0.9 / (th * (0.9 - th)), rel=1e-6)

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


def test_msmr_explicit_ds_matches_baker_verbrugge(tmp_path):
    """An MSMR material given Ds(theta) from the Baker-Verbrugge law runs as one given Dsref."""
    d = json.loads(MSMR.read_text())
    f = d["F"] / (d["R"] * d["Tref"])
    for e in ("neg", "pos"):
        m = d["regions"][e]["materials"][0]
        U0, X, w = (np.asarray(m[k]["value"], float)[:, None] for k in ("U0", "X", "omega"))
        U = np.linspace(U0.min() - 1, U0.max() + 1, 20001)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):    # far gallery tails
            g = np.exp(f * (U - U0) / w)
            theta = (X / (1 + g)).sum(0)
            dthdU = -(f * X * g / w / (1 + g) ** 2).sum(0)
            ds = m.pop("Dsref")["value"] * f * theta * (1 - theta) / -dthdU
        keep = np.isfinite(ds) & (np.diff(theta, prepend=np.inf) < 0)       # theta falls with U
        m["Ds"] = {"type": "LUT", "Ea": 0, "scale": 1,
                   "value": {"x": theta[keep][::-1].tolist(), "y": ds[keep][::-1].tolist()}}
    bv, _ = discharge(load_spec(MSMR), t_on=300, t_end=600)
    explicit, _ = discharge(spec_from(d, tmp_path), t_on=300, t_end=600)
    assert explicit["termination"] == "final time"
    np.testing.assert_allclose(explicit["Vcell"], bv["Vcell"], atol=2e-3)


def test_bv_kinetics_on_msmr_galleries_match_their_exchange_current(tmp_path):
    """
    Butler-Volmer on an MSMR material, with k0(theta) the galleries' total exchange
    current over theta^a*(1-theta)^(1-a), runs as the galleries do (one alpha for all).
    """
    d = json.loads(MSMR.read_text())
    f = d["F"] / (d["R"] * d["Tref"])
    for e in ("neg", "pos"):
        m = d["regions"][e]["materials"][0]
        m["alpha"] = {"type": "vector", "Ea": 0, "scale": 1, "value": [0.5] * len(m["U0"]["value"])}
    bv = copy.deepcopy(d)
    for e in ("neg", "pos"):
        m = bv["regions"][e]["materials"][0]
        U0, X, w, k0 = (np.asarray(m[k]["value"], float)[:, None] for k in ("U0", "X", "omega", "k0"))
        U = np.linspace(U0.min() - 1, U0.max() + 1, 20001)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            g = np.exp(f * (U - U0) / w)
            x = X / (1 + g)
            theta = x.sum(0)
            i0 = (k0 * x ** w * np.exp(0.5 * (U - U0) * f)).sum(0)
            k = i0 / np.sqrt(theta * (1 - theta))
        keep = np.isfinite(k) & (k > 0) & (np.diff(theta, prepend=np.inf) < 0)
        m["k0"] = {"type": "LUT", "Ea": 0, "scale": 1, "value": {"x": theta[keep][::-1].tolist(), "y": k[keep][::-1].tolist()}}
        m["alpha"] = {"type": "scalar", "Ea": 0, "scale": 1, "value": 0.5}
        bv["regions"][e]["kinetics"] = ["BV"]
    galleries, _ = discharge(spec_from(d, tmp_path, "msmr.json"), t_on=300, t_end=600)
    butler, _ = discharge(spec_from(bv, tmp_path, "bv.json"), t_on=300, t_end=600)
    assert butler["termination"] == "final time"
    np.testing.assert_allclose(butler["Vcell"], galleries["Vcell"], atol=1e-3)


@pytest.mark.parametrize("path", [NMC30, MSMR], ids=["standard", "msmr"])
def test_stretch_of_one_is_plain_baker_verbrugge(path, tmp_path):
    d = json.loads(path.read_text())
    for e in ("neg", "pos"):
        d["regions"][e]["materials"][0]["mD"] = {"type": "scalar", "Ea": 0, "scale": 1, "value": 1.0}
    plain, _ = discharge(load_spec(path), t_on=300, t_end=600)
    stretched, _ = discharge(spec_from(d, tmp_path), t_on=300, t_end=600)
    np.testing.assert_allclose(stretched["Vcell"], plain["Vcell"], atol=1e-5)


def test_files_without_kinetics_choose_from_k0(tmp_path):
    """A file that predates the kinetics list: galleries with a k0 each are MSMR, one k0 is BV."""
    d = json.loads(MSMR.read_text())
    for e in ("neg", "pos"):
        d["regions"][e].pop("kinetics", None)
    model = FOMModel(spec_from(d, tmp_path), 0.5, 298.15, Options(mesh=5))
    assert [model._kinetics(e, 0) for e in ("neg", "pos")] == ["MSMR", "MSMR"]
    for k in ("k0", "alpha"):
        d["regions"]["neg"]["materials"][0][k] = {"type": "scalar", "Ea": 0, "scale": 1, "value": 0.5}
    assert FOMModel(spec_from(d, tmp_path, "one.json"), 0.5, 298.15, Options(mesh=5))._kinetics("neg", 0) == "BV"


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



def test_soc_limit_ends_the_step_and_the_next_follows():
    # With step numbers, the SOC limit ends the charge step; the rest step runs at once.
    spec = load_spec(NMC30)
    I = -2.0 * capacity(spec, "neg")
    t = np.arange(0.0, 4300 + 1, 10.0)
    step = np.where(t < 4000, 1, 2)
    sim = {"time": t, "Iapp": np.where(step == 1, I, 0.0), "T": 25.0, "SOC0": 20, "step": step}
    out = simulate(spec, sim, RunOptions(model=Options(mesh=4, limits=Limits(SOCLimit=0.7))))
    hit = np.flatnonzero(out["SOC"] >= 0.7 - 1e-6)[0]
    assert out["termination"] == "final time"
    assert out["SOC"].max() == pytest.approx(0.7, abs=1e-4)     # the double layer settles into the rest
    assert out["time"][-1] == pytest.approx(out["time"][hit] + 300, abs=1)   # the rest follows at once
    assert np.all(np.abs(out["Iapp"][out["time"] > out["time"][hit] + 1]) < 1e-6)


@pytest.mark.parametrize("thetae_max", [3.0, 2.2], ids=["plating", "thetae-then-plating"])
def test_a_limit_engaging_mid_step_holds_and_continues(thetae_max):
    # cellSKON603 at 2.5C from 10 %: the plating limit engages inside a step,
    # with the current still at its demand; the run must hold the limit and go on.
    # At 2.2 the theta_e limit holds first, then hands over to the plating limit
    # while theta_e sits at its bound: both must keep holding.
    spec = load_spec(SKON)
    lim = Limits(phiseMin=0.01, thetaeMin=0.1, thetaeMax=thetae_max, SOCLimit=0.8)
    out, _ = discharge(spec, rate=-2.5, t_on=1800, t_end=1800, soc0=10,
                       opts=RunOptions(model=Options(mesh=4, limits=lim)))
    k1 = np.argmin(np.abs(out["xLocs"]["Phise"] - 1))
    assert out["termination"] != "failed", out["message"]
    assert out["time"][-1] > 700
    assert out["Phise"][:, k1].min() >= 0.01 - 1e-5
    assert out["Thetae"].max() <= 1.01 * thetae_max      # its controller overshoots briefly as it engages



def test_chunk_restarts_skip_casadi_initialisation(monkeypatch):
    # A constant current over 3 chunks: only the first step solves for consistent states.
    calls = []
    solve = pybamm.CasadiAlgebraicSolver._integrate
    monkeypatch.setattr(pybamm.CasadiAlgebraicSolver, "_integrate",
                        lambda self, *a, **k: calls.append(1) or solve(self, *a, **k))
    out, _ = discharge(load_spec(NMC30), t_on=2100, t_end=2100)
    assert out["time"].size > 2 * CHUNK and out["termination"] == "final time"
    assert len(calls) == 1



def test_a_thetae_limit_only_cuts_the_current():
    # A theta_e limit well below the charge's own swing holds for most of it: the
    # current stays between its demand and zero, never past it and never reversed.
    lim = Limits(thetaeMax=1.5, SOCLimit=0.8)
    out, I = discharge(load_spec(SKON), rate=-2.5, t_on=900, t_end=900, soc0=10,
                       opts=RunOptions(model=Options(mesh=4, limits=lim)))
    assert out["termination"] != "failed", out["message"]
    assert np.all(out["Iapp"] >= I - 1e-6) and np.all(out["Iapp"] <= 1e-6)
    assert np.any(out["Iapp"] > 0.5 * I)                           # it did cut

def test_a_restart_casadi_cannot_initialise_uses_idaklu():
    class Stub:
        def __init__(self, err=None):
            self.err = err

        def step(self, *a, **k):
            if self.err:
                raise pybamm.SolverError(self.err)
            return "idaklu"

    casadi = "Could not find consistent states: rootfinder process failed"
    assert _step(Stub(casadi), Stub(), object(), False, None, 1.0, {}, None) == "idaklu"
    with pytest.raises(pybamm.SolverError):                   # the first step has no carried state
        _step(Stub(casadi), Stub(), None, False, None, 1.0, {}, None)
    with pytest.raises(pybamm.SolverError):                   # other failures are not retried
        _step(Stub("Events ['limit Vmax'] are non-positive"), Stub(), object(), False, None, 1.0, {}, None)


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


def test_element_counts_override_the_preset():
    spec = load_spec(NMC30)
    m = FOMModel(spec, 0.5, 298.15, Options(mesh=3, elements={"sep": 7, "particle": 120}))
    pts = {k.name: v for k, v in m.default_var_pts.items()}
    assert pts["x_sep"] == 7 and pts["x_neg"] == pts["x_pos"] == 10       # preset 3: 10 per region
    assert all(v == 120 for k, v in pts.items() if k.startswith("r_"))
    with pytest.raises(ValueError):
        Options(elements={"particle": 1})
