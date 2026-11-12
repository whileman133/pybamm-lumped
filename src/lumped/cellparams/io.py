"""
lumped.cellparams.io

Utilities for reading/writing cell models from/to disk.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .model import Parameter, Section, LUT, CellParams


def load_cell_params(filename):
    """
    Load cell parameter values from xlsx file. Returns a CellModel object.
    """

    df = pd.read_excel(filename, sheet_name='Parameters', header=None)
    n = df.shape[0]
    col_sec = df[0]
    col_codename = df[1]
    col_value = df[2]
    col_Eact = df[3]
    col_unit = df[4]

    # Determine start of parameter sections.
    ind_sec, = np.where(col_sec.str.startswith('#') == True)
    names_sec = col_sec[ind_sec].str.lstrip('#')
    ind_sec_params = ind_sec + 3  # skip two header rows

    # Collect parameter data by section.
    param_dict = defaultdict(list)
    for i_sec, name_sec in zip(ind_sec_params, names_sec):
        l = 0
        while i_sec+l < n and not (isinstance(col_codename[i_sec+l], float) and np.isnan(col_codename[i_sec+l])):
            name = col_codename[i_sec+l]
            value = col_value[i_sec+l]
            Eact = col_Eact[i_sec+l]
            unit = col_unit[i_sec+l]
            param_dict[name_sec].append({
                "name": name,
                "value": value,
                "Eact": Eact,
                "unit": unit,
            })
            l += 1

    # Process table references and vectors.
    for name_sec, params in param_dict.items():
        for param in params:
            if isinstance(param['value'], str) and param['value'].startswith('#'):
                sheet_name = param['value'].lstrip('#')
                tab = pd.read_excel(filename, sheet_name=sheet_name, header=None)
                x, y = tab[0], tab[1]
                param['value'] = LUT(x.to_numpy(), y.to_numpy())
            elif isinstance(param['value'], str) and param['value'].startswith('['):
                value = str2value(param, 'value')
                if len(value) == 1:
                    param['value'] = value[0]
                else:
                    param['value'] = value
            if isinstance(param['Eact'], str) and param['Eact'].startswith('['):
                Eact = str2value(param, 'Eact')
                if len(Eact) == 1:
                    param['Eact'] = Eact[0]
                else:
                    param['Eact'] = Eact

            # Convert kJ to J.
            param['Eact'] *= 1_000

    # Build cell model.
    gen = param_dict.pop('general')
    model = CellParams(gen)
    for name_sec, params in param_dict.items():
        sec = Section(name_sec, secname2kind(name_sec))
        for param in params:
            param_obj = Parameter(param['name'], param['unit'], param['value'], param['Eact'])
            setattr(sec, param['name'], param_obj)
        setattr(model, name_sec, sec)

    return model


def str2value(param, field='value'):
    try:
        value = json.loads(param[field])
    except json.decoder.JSONDecodeError:
        # Also accept MATLAB-style vectors, e.g. '[0.1  0.2 0.3]'.
        try:
            value = [float(v) for v in param[field].strip('[]').replace(',', ' ').split()]
        except ValueError:
            raise ValueError(f"Could not decode parameter '{param['name']}' as a JSON or MATLAB vector.")
    if not isinstance(value, list):
        raise ValueError(f"Expected list for parameter '{param['name']}' but got '{type(value)}'.")
    value = np.array(value)
    if value.dtype.kind == 'U':
        raise ValueError(f"Expected numeric type for parameter '{param['name']}' but got string(s).")
    if len(value) == 0:
        raise ValueError(f"Parameter '{param['name']}' has no value.")
    return value


def secname2kind(secname):
    if secname.lower() == 'general':
        return 'Meta'
    elif secname.lower() == 'const':
        return 'Global'
    elif secname.lower() in ['neg', 'pos']:
        return 'Electrode3D'
    elif secname.lower() == 'sep':
        return 'ElectrolyteLayer'
    else:
        raise ValueError(f"Don't know how to classify section '{secname}'.")

