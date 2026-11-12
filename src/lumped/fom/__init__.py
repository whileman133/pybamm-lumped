"""
lumped.fom

The BMS3-toolbox full-order model in PyBaMM, built from a compiled cell model
that the toolbox exports as JSON (exportLumpedModel.m).
"""
from .spec import CellSpec, Param, SpecError, load_spec

__all__ = ["CellSpec", "Param", "SpecError", "load_spec"]
