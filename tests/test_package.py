"""
Packaging and utility tests.
"""
import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid

import lumped

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"


def test_lib_defaults_match_fixture():
    # The packaged defaults are what scripts/make-lib-defaults.py writes from the fixture.
    spec = importlib.util.spec_from_file_location("make_lib_defaults", ROOT / "scripts" / "make-lib-defaults.py")
    make_lib_defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_lib_defaults)
    packaged = json.loads((files("lumped") / "lib-defaults.json").read_text())
    assert packaged == json.loads(json.dumps(make_lib_defaults.build()))


def test_lmb_model_requires_positive_J():
    with pytest.raises(ValueError):
        lumped.LumpedLMBModel(J=0)


def test_sim_drive_scales_to_requested_soc_window(tmp_path):
    cell = lumped.load_cell_params(DATA / "cellNMC30.xlsx")
    t = np.arange(0.0, 601.0, 10.0)
    i = 1.0 + 0.5 * np.sin(2 * np.pi * t / 120)
    csv = tmp_path / "cycle.csv"
    np.savetxt(csv, np.column_stack([t, i]), delimiter=",")

    sim = lumped.sim_drive(cell, csv, soc0=60, socf=55)

    time = sim.solution["time [s]"].entries
    iapp = sim.solution["iapp [A]"].entries
    QdisAh = trapezoid(iapp, time) / 3600
    np.testing.assert_allclose(QdisAh, 0.05 * cell.const.Q(), rtol=1e-3)
    pos_soc = sim.solution["pos_soc [%]"].entries
    assert pos_soc[-1] == pytest.approx(55, abs=0.5)


@pytest.mark.parametrize("text, expected", [
    ("[0.1, 0.2, 0.3]", [0.1, 0.2, 0.3]),     # JSON
    ("[0.1  0.2 0.3]", [0.1, 0.2, 0.3]),      # MATLAB, as in cellNMCMSMR.xlsx
])
def test_str2value_vector_formats(text, expected):
    from lumped.cellparams.io import str2value
    np.testing.assert_allclose(str2value({"name": "p", "value": text}), expected)


def test_str2value_rejects_garbage():
    from lumped.cellparams.io import str2value
    with pytest.raises(ValueError, match="Could not decode"):
        str2value({"name": "p", "value": "[a b]"})
