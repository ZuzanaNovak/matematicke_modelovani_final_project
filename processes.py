import numpy as np
from utils import tanh
from params import Params

#  helpers 
def concentrations(state, P: Params):
    x1,u1p,u2p,u11,u12,u13,u2 = state
    y1 = x1 / P.V1 * 100.0
    y2 = u11 / P.V11
    y3 = u12 / P.V12
    y4 = u13 / P.V13
    y5 = u2  / P.V2
    return y1,y2,y3,y4,y5

def deviations(y1,y2,y3,y4,y5,P):
    e1  = y1 - P.y1_bas
    e12 = y3 - P.y3_bas
    e13 = y4 - P.y4_bas
    e21 = y5 - P.y5_bas
    return e1,e12,e13,e21

def scale_bw(rate_mg_min_per_kg: float, P: Params) -> float:
    return rate_mg_min_per_kg * P.BW_kg

#   glucose unit processes (per kg)
def F1_LiverProduction_perkg(x1,u12,u2,P: Params):
    y1,_,y3,_,y5 = concentrations((x1,0,0,0,u12,0,u2),P)
    e1,e12,_,e21 = deviations(y1,0,y3,0,y5,P)
    G1 = 0.5*(1 + tanh(P.b11*(e21 + P.c11)))
    H1 = 0.5*(1 - tanh(P.b12*(e12 + P.c12)))
    M1 = 0.5*(1 - tanh(P.b13*(e1  + P.c13)))
    return P.a11 * G1 * H1 * M1

def F2_LiverUptake_perkg(x1,u12,P: Params):
    y1,_,y3,_,_ = concentrations((x1,0,0,0,u12,0,0),P)
    e1,e12,_,_ = deviations(y1,0,y3,0,0,P)
    H2 = 0.5*(1 + tanh(P.b21*(e12 + P.c21)))
    M2 = P.a221*(1 + P.a222*0.5*(1 + tanh(P.b22*(e1 + P.c22))))
    return P.a21 * H2 * M2

def F3_RenalExcretion_perkg(x1,P: Params):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    M31 = 0.5*(1 + tanh(P.b31*(y1 + P.c31)))
    M32 = P.a321*y1 + P.a322
    return max(0.0, M31*M32)

def F4_PeripheralID_perkg(x1,u13,P: Params):
    y1,_,_,y4,_ = concentrations((x1,0,0,0,0,u13,0),P)
    e1,_,e13,_ = deviations(y1,0,0,y4,0,P)
    H3 = 0.5*(1 + tanh(P.b33*(e13 + P.c33)))
    M3 = 0.5*(1 + tanh(P.b32*(e1  + P.c32)))
    return 5.5e2 * (P.a31 * H3 * M3)

def F5_PeripheralInd_perkg(x1,P: Params):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    M41 = P.a41 * 0.5*(1 + tanh(P.b41*(e1 + P.c41)))
    M42 = P.a42 * 0.5*(1 + tanh(P.b42*(e1 + P.c42)))
    return 3e2 * (M41 + M42)

#   secretions  
def W_insulin_synthesis(x1,P: Params) -> float:
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    return 0.5*P.a5*(1 + tanh(P.b5*(e1 + P.c5)))

def F6_secretion(x1,u2p,P: Params) -> float:
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    return 0.5*P.a6*(1 + tanh(P.b6*(e1 + P.c6))) * u2p

def F7_glucagon_secretion(x1, u13, P: Params) -> float:
    """α-cell glucagon secretion (pg/min), with tonic + floor + max applied."""
    y1,_,_,y4,_ = concentrations((x1,0,0,0,0,u13,0), P)
    e1,_,e13,_ = deviations(y1,0,0,y4,0,P)

    # inhibitions
    H7 = 0.5*(1 - tanh(P.b71*(e13 + P.c71)))   # insulin effect
    M7 = 0.5*(1 - tanh(P.b72*(e1  + P.c72)))   # glucose effect

    # tonic + dynamic
    F7 = P.F7_tonic + P.a7 * H7 * M7

    # enforce floor and cap
    if P.F7_max is None:
        P.F7_max = P.F7_max_perkg * P.BW_kg
    return float(np.clip(F7, P.F7_floor, P.F7_max))


def solve_glucagon_gain(P: Params) -> float:
    # Basal inhibitions (at deviations = 0)
    H7b = 0.5*(1 - np.tanh(P.b71*(0.0 + P.c71)))
    M7b = 0.5*(1 - np.tanh(P.b72*(0.0 + P.c72)))
    HMb = max(H7b * M7b, 1e-12)

    # Basal secretion needed to keep y5 steady: clearance × amount
    F7_basal = P.h01 * P.y5_bas * P.V2  # pg/min

    # Set tonic piece and max ceiling
    P.F7_tonic = float(P.F7_tonic_frac) * F7_basal
    P.F7_max   = float(P.F7_max_perkg) * P.BW_kg

    # Dynamic gain supplies the rest at basal
    dynamic_target = max(F7_basal - P.F7_tonic, 0.0)
    a7 = dynamic_target / HMb

    # If basal would exceed the cap, scale both down proportionally
    if (P.F7_tonic + a7*HMb) > P.F7_max:
        scale = P.F7_max / (P.F7_tonic + a7*HMb)
        P.F7_tonic *= scale
        a7 *= scale

    P.a7 = a7
    return a7
def F7_target(x1: float, u13: float, P: Params) -> float:
    y1,_,_,y4,_ = concentrations((x1,0,0,0,0,u13,0), P)
    e1,_,e13,_  = deviations(y1,0,0,y4,0,P)
    H7 = 0.5*(1 - tanh(P.b71*(e13 + P.c71)))     # insulin inhibition
    M7 = 0.5*(1 - tanh(P.b72*(e1  + P.c72)))     # glucose inhibition
    F7 = P.F7_tonic + P.a7 * H7 * M7
    if P.F7_max is None:
        P.F7_max = P.F7_max_perkg * P.BW_kg
    return float(np.clip(F7, P.F7_floor, P.F7_max))

def solve_insulin_basal(P: Params, u1p0: float, u2p0: float) -> Params:
    q = (2.0*P.k01*u1p0)/(P.a6*u2p0) - 1.0  # tanh(b6*c6)
    q = float(np.clip(q, -0.999, 0.999))
    P.c6 = (np.arctanh(q))/P.b6
    W0 = P.k01*u1p0 - P.k21*u2p0
    g  = 0.5*(1.0 + np.tanh(P.b5*(0.0 + P.c5)))
    P.a5 = W0 / max(g, 1e-12)
    return P
