"""
lumped.comsol

Load COMSOL simulation output exported to MATLAB .mat files.
"""
from dataclasses import dataclass

import numpy as np
import scipy.io as sio


@dataclass
class FOMOutput:
    ts: float
    time: np.ndarray
    iapp: np.ndarray
    vcell: np.ndarray
    thetass: np.ndarray       # over pos electrode
    thetae: np.ndarray
    xthetass: np.ndarray
    xthetae: np.ndarray


def load_mat_comsol_sim(filepath):
    """
    Load COMSOL simulation output from .mat file.
    """

    simData = sio.loadmat(filepath)
    time = simData['simData']['time'][0, 0].T[0]
    ts = float(np.mean(np.diff(time)))
    iapp = simData['simData']['Iapp'][0, 0].T[0]
    vcell = simData['simData']['Vcell'][0, 0].T[0]
    thetass = simData['simData']['Thetass'][0, 0]
    thetae = simData['simData']['Thetae'][0, 0]
    xthetass = simData['simData']['xLocs'][0, 0]['Thetass'][0, 0].T[0]
    xthetae = simData['simData']['xLocs'][0, 0]['Thetae'][0, 0].T[0]

    return FOMOutput(ts, time, iapp, vcell, thetass, thetae, xthetass, xthetae)


def load_mat_comsol_eis_sim(filepath):
    """
    Load COMSOL EIS simulation output from .mat file.
    """

    simData = sio.loadmat(filepath)
    socPct = simData['SOCs'][0, 0]*100
    freq = simData['freqs'][0]
    Zcell = simData['Z'].T[0]

    return socPct, freq, Zcell
