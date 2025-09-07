# state.py
import numpy as np
from params import Params

def initial_state(P: Params):
    x1_0 = (P.y1_bas/100.0) * P.V1
    u11_0 = P.y2_bas * P.V11
    u12_0 = P.y3_bas * P.V12
    u13_0 = P.y4_bas * P.V13
    u2_0  = P.y5_bas * P.V2
    # pancreatic pools
    u1p_0 = 4.9e6
    u2p_0 = 4.9e5
    return np.array([x1_0,u1p_0,u2p_0,u11_0,u12_0,u13_0,u2_0], float)

from processes import solve_glucagon_gain, F7_target

def initial_state(P: Params):
    x1_0 = (P.y1_bas/100.0) * P.V1
    u11_0 = P.y2_bas * P.V11
    u12_0 = P.y3_bas * P.V12
    u13_0 = P.y4_bas * P.V13
    u2_0  = P.y5_bas * P.V2
    u1p_0 = 4.9e6
    u2p_0 = 4.9e5

    # alpha-cell smoothed output starts at basal target
    return np.array([x1_0,u1p_0,u2p_0,u11_0,u12_0,u13_0,u2_0], dtype=float)

