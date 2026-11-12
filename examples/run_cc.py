"""
run_cc.py

Simulate a 1C constant-current discharge from 100% to 80% SOC, then a 10-minute rest.
Run with: uv run examples/run_cc.py [cell.xlsx]
The cell defaults to the cellNMC30 test fixture.
"""
import sys
from pathlib import Path

import lumped

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "data" / "cellNMC30.xlsx"

if __name__ == '__main__':
    params = lumped.load_cell_params(sys.argv[1] if len(sys.argv) > 1 else FIXTURE)
    Q = params.const.Q()
    sim = lumped.sim_cc(params, 1*Q, 100, 80, t_rest=10*60)
    sim.plot()
