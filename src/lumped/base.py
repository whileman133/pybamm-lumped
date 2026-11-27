"""
lumped.base

Base definitions for lumped-parameter models.
"""

import pybamm


def msmr_exchange_current(k0, X, omega, alpha, Uss, U0, f):
    """
    MSMR exchange current of one gallery without its electrolyte factor thetae^(1-alpha).

    Parameters
    ----------
    k0, X, omega, alpha : pybamm.Symbol or float
        Gallery rate constant [A], size, width and charge-transfer coefficient.
    Uss : pybamm.Symbol
        Surface potential of the particle. [V]
    U0 : pybamm.Symbol or float
        Gallery equilibrium potential. [V]
    f : pybamm.Symbol or float
        F/(R T). [1/V]

    Returns
    -------
    i0 : pybamm.Symbol
        k0 * x^(omega*alpha) * (X - x)^(omega*(1-alpha)), x = X/(1 + exp(z)). [A]

    Notes
    -----
    With z = f*(Uss - U0)/omega the form is k0 * X^omega * (e^(alpha z) + e^(-(1-alpha) z))^(-omega).
    - The log-sum-exp keeps the exponent at most omega*ln(X): no overflow, underflow or floor.
    - x and X - x are never formed, so neither end of the gallery is singular.
    - d(ln i0)/dUss = f*((1 - alpha) - sigmoid(z)) is bounded.
    """
    z = f * (Uss - U0) / omega
    lse = pybamm.maximum(alpha * z, -(1 - alpha) * z) + pybamm.log(1 + pybamm.exp(-abs(z)))
    return k0 * pybamm.exp(omega * (pybamm.log(X) - lse))


class _Container:
    """
    Generic container class for parameters and variables.
    """

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            raise AttributeError(f"Cannot find {item} in model.")

    def __setattr__(self, key, value):
        return super().__setattr__(key, value)


class BaseLumpedModel(pybamm.BaseModel):
    """
    Base class for lumped-parameter models.
    """

    def __init__(self, name="Unnamed lumped-parameter model"):
        super().__init__(name)
        self.param = _Container()    # model parameters
        self.var = _Container()      # model variables