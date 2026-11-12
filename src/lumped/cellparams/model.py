"""
lumped.cellparams.model

Objects for working with cell models.
"""

from dataclasses import dataclass
from numbers import Number
from typing import Union

import numpy as np
import scipy.constants as const


class _Container:
    """
    Generic container class for parameters.
    """

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            raise AttributeError(f"Cannot find {item}.")

    def __setattr__(self, key, value):
        return super().__setattr__(key, value)


@dataclass
class LUT:
    """
    Lookup table.
    """
    x: np.ndarray
    y: np.ndarray

    def __call__(self, x, **kwargs):
        return np.interp(x, self.x, self.y)


@dataclass
class Parameter:
    name: str
    unit: str
    value: Union[LUT, np.ndarray, Number]
    Eact: Union[np.ndarray, Number]

    def __call__(self, *args, TdegC: float = 25.0, **kwargs):
        """
        Fetch the value of this parameter.
        """
        if isinstance(self.value, LUT):
            value_ref = self.value(*args, **kwargs)
        else:
            value_ref = self.value

        # Model temperature dependence with Arrhenius relation
        Tref = const.zero_Celsius + 25.0
        T = const.zero_Celsius + TdegC
        value = value_ref * np.exp(self.Eact / const.R * (1/Tref - 1/T))
        return value

class Section(_Container):
    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self._param_names = set()

    def __setattr__(self, key, value):
        v = super().__setattr__(key, value)
        if isinstance(value, Parameter):
            self._param_names.add(key)
        return v

    @property
    def parameter_names(self):
        return tuple(self._param_names)


class CellParams(_Container):
    def __init__(self, params):
        for param in params:
            setattr(self, f"{param['name']}", param['value'])
        self._sec_names = set()

    def __setattr__(self, key, value):
        v = super().__setattr__(key, value)
        if isinstance(value, Section):
            self._sec_names.add(key)
        return v

    @property
    def section_names(self):
        return tuple(self._sec_names)

    def __str__(self):
        return str(vars(self))