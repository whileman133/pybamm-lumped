"""
lumped.sim

Simulation utilities for the lumped-parameter LIB model.
"""
import math

import numpy as np
import pybamm
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator

from ._vendor import pybammeis
from .cellparams.model import CellParams
from .libcell import LumpedLIBModel
from .libparams import get_initial, get_lib_param_values, lib_model_options


#
# PyBaMM simulation utilities
#


def sim_eis(
    cell: CellParams,
    freq: np.ndarray,
    soc0: float,
    TdegC: float = 25.0,
):
    """
    Solve the PDE model for EIS.

    :param cell: LMB cell model object containing parameter values.
    :param freq: Frequency vector [Hz].
    :param soc0: Initial cell SOC [%].
    :param TdegC: Cell temperature [degC]. DEFAULT 25.
    :return: PyBaMM-EIS Simulation object
    """

    T = TdegC + 273.15

    # Construct PyBaMM model.
    pde_model = LumpedLIBModel(**lib_model_options(cell))

    # Collect parameter values.
    param0 = get_initial(cell, soc0, TdegC, pde_model)
    param = get_lib_param_values(cell, pde_model, TdegC)
    param.update({
        "T [K]": T,
        "neg_thetas0": param0.neg_thetas,
        "neg_Uocp0 [V]": param0.neg_Uocp,
        "pos_thetas0": param0.pos_thetas,
        "pos_Uocp0 [V]": param0.pos_Uocp,
        "iapp [A]": 0,
    }, check_already_exists=False)

    # Run simulation.
    sim = pybammeis.EISSimulation(pde_model, parameter_values=param)
    sim.solve(freq)

    return sim


def sim_profile(
    cell: CellParams,
    time: np.ndarray,
    iapp: np.ndarray,
    soc0: float,
    TdegC: float = 25.0,
):
    """
    Solve the PDE model for the specified parameter values and current profile.

    :param cell: LMB cell model object containing parameter values.
    :param time: Vector of time values [s].
    :param iapp: Vector of applied current values [A].
    :param soc0: Initial cell SOC [%].
    :param TdegC: Cell temperature [degC]. DEFAULT 25.
    :return: PyBaMM Simulation object
    """

    T = TdegC + 273.15

    # Construct PyBaMM model.
    pde_model = LumpedLIBModel(**lib_model_options(cell))

    # Collect parameter values.
    param0 = get_initial(cell, soc0, TdegC, pde_model)
    param = get_lib_param_values(cell, pde_model, TdegC)
    param.update({
        "T [K]": T,
        "iapp [A]": pybamm.Interpolant(
            time, iapp, pybamm.t, interpolator="linear"
        ),
        "neg_thetas0": param0.neg_thetas,
        "neg_Uocp0 [V]": param0.neg_Uocp,
        "pos_thetas0": param0.pos_thetas,
        "pos_Uocp0 [V]": param0.pos_Uocp,
    }, check_already_exists=False)

    # Run simulation.
    sim = pybamm.Simulation(pde_model, parameter_values=param)
    sim.solve(t_eval=time)

    return sim


def sim_cc(
    cell: CellParams,
    i_galv: float,
    soc0: float,
    socf: float,
    TdegC: float = 25.0,
    ts: float = 1.0,
    t_rest: float = 0,
):
    """
    Solve the PDE model for a constant-current dis/charge profile.

    :param cell:
    :param i_galv: Constant dis/charge current [A].
    :param soc0: Initial SOC [%].
    :param socf: Final SOC [%].
    :param TdegC: Cell temperature [degC]. DEFAULT 25.
    :param ts: Sampling interval [s]. DEFAULT 1.0.
    :return: PyBaMM Simulation object
    """

    i_galv = abs(i_galv)  # ignore sign, can infer from soc0 and socf

    # Total amount of charge to remove from (+) / add to (-) the cell [Ah].
    QdisAh = (soc0 - socf) / 100 * cell.const.Q(TdegC=TdegC)

    # Total amount of time to spend dis/charging at i_galv [s].
    t_cc = 3600 * abs(QdisAh) / i_galv

    # Build time and applied current vectors.
    time = np.arange(0, t_cc + t_rest + ts, ts)
    iapp = np.zeros_like(time)
    iapp[time <= t_cc] = np.sign(soc0 - socf)*i_galv

    # Run the simulation.
    return sim_profile(cell, time, iapp, soc0, TdegC)


def sim_gitt(
    cell: CellParams,
    t_galv: float,
    t_rest: float,
    i_galv: float,
    soc0: float,
    socf: float,
    TdegC: float = 25.0,
    ts: float = 1.0,
):
    """
    Solve the PDE model for a galvanostatic intermittent titration
    technique (GITT) current profile.

    :param cell: LMB cell model object containing parameter values.
    :param t_galv: Dis/charge interval duration [s].
    :param t_rest: Rest interval duration [s].
    :param i_galv: Current magnitude during dis/charge intervals [A].
    :param soc0: Initial SOC of the cell [%].
    :param socf: Final SOC of the cell [%].
    :param TdegC: Temperature of the cell [degC]. DEFAULT 25.
    :param ts: Sampling interval [s]. DEFAULT 1.0.
    :return: PyBaMM Simulation object
    """

    i_galv = abs(i_galv)  # ignore sign, can infer from soc0 and socf

    # Total amount of charge to remove from (+) / add to (-) the cell [Ah].
    QdisAh = (soc0 - socf)/100 * cell.const.Q(TdegC=TdegC)

    # Total amount of time to spend dis/charging at i_galv [s].
    t_dis = 3600 * abs(QdisAh) / i_galv

    # Number of required dis/charge intervals.
    n_dis = math.ceil(t_dis/t_galv)

    # Total duration of the GITT profile.
    t_gitt = n_dis * (t_galv + t_rest)

    # Build time and applied current vectors.
    time = np.arange(0, t_gitt + ts, ts)
    iapp = np.zeros_like(time)
    for k in range(n_dis):
        t_start = k * (t_galv + t_rest)
        ind = np.logical_and(t_start <= time, time <= t_start + t_galv)
        iapp[ind] = np.sign(soc0 - socf) * i_galv

    # Run the simulation.
    return sim_profile(cell, time, iapp, soc0, TdegC)


def sim_drive(
    cell: CellParams,
    cycle_csv,
    soc0: float,
    socf: float,
    TdegC: float = 25.0,
    ts: float = 1.0,
):
    """
    Solve the PDE model for a dynamic drive current profile.

    :param cell: LIB cell model object containing parameter values.
    :param cycle_csv: Path to a two-column CSV file (time [s], current [A]) holding the drive cycle.
        The current is rescaled so the cycle moves the cell from soc0 to socf.
    :param soc0: Initial cell soc [%].
    :param socf: Final cell soc [%].
    :param TdegC: Cell temperature. DEFAULT 25.
    :param ts: Sampling interval [s]. DEFAULT 1.0.
    :return: PyBaMM Simulation object
    """

    # Load drive cycle from file.
    data = np.genfromtxt(cycle_csv, delimiter=',')
    time0 = data[:, 0]
    iapp0 = data[:, 1]

    # Interpolate over unform time variable.
    time = np.arange(np.min(time0), np.max(time0) + ts, ts)
    iapp = PchipInterpolator(time0, iapp0)(time)

    # Rescale applied current.
    QdisAh0 = trapezoid(iapp, time)/3600
    QdisAh = (soc0 - socf)/100 * cell.const.Q(TdegC=TdegC)
    iapp = iapp*QdisAh/QdisAh0

    # Run the simulation.
    return sim_profile(cell, time, iapp, soc0, TdegC)
