# Vendored: pybammeis 0.1.5

Upstream: <https://github.com/pybamm-team/pybamm-eis> (PyPI `pybammeis==0.1.5`), MIT
license, see `LICENSE.md`.

Local modifications (why this is vendored instead of a dependency):

- `eis_simulation.py` adds `FunctionControlAlg` and uses the lumped models' variable
  names (`iapp [A]`, `iapp variable [A]`, `vcell [V]`) instead of PyBaMM's
  (`Current [A]`, `Voltage [V]`). The initial voltage guess is
  `pos_Uocp0 - neg_Uocp0`, so EIS currently supports `LumpedLIBModel` only.
- Imports made package-relative so it can live under `lumped._vendor`.
