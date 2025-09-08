import math
from dataclasses import dataclass

@dataclass
class Params:
    BW_kg: float = 70.0
    # volumes (ml) filled later in set_volumes()
    V1: float=None; V11: float=None; V12: float=None; V13: float=None; V2: float=None

    # cohort basals
    y1_bas: float = 77.0   # mg/dl (glucose)
    y2_bas: float = 7.0    # µU/ml (plasma insulin)
    y3_bas: float = 7.0    # µU/ml (portal/liver insulin)
    y4_bas: float = 7.0    # µU/ml (interstitial insulin)
    y5_bas: float = 120.0  # pg/ml (glucagon)

    # beta-cell pools/secretion
    k01: float = 0.03; k21: float = 4.34e-3; k12: float = 0.0
    a5: float = 0.287; b5: float = 1.51e-2; c5: float = -92.3
    a6: float = 1.4;   b6: float = 9.23e-3; c6: float = -10.0

    # insulin kinetics
    m01: float = 0.110; m02: float = 0.170; m12: float = 0.209
    m13: float = 0.018; m21: float = 0.26;  m31: float = 0.060

    # glucagon kinetics
    h01: float = math.log(2)/8.0

    # glucagon secretion 
    a7:  float = None
    b71: float = 0.0020; c71: float = 150.0
    b72: float = 0.030;  c72: float = 80.0

    # Hepatic production F1 (per kg)
    a11: float = 5.5;  b11: float = 1.60; c11: float = -1.10
    b12: float = 0.784; c12: float = -108.5
    b13: float = 0.0275; c13: float = 26.0

    # Hepatic uptake F2 (per kg)
    a21: float = 0.5e-3
    b21: float = 0.10; c21: float = 10.0
    a221: float = 60.0; a222: float = 1.00
    b22: float = 0.090; c22: float = -90.0

    # Renal F3 (per kg)
    b31: float = 0.02; c31: float = -190.0
    a321: float = 1.43e-5; a322: float = -1.31e-5

    # Peripheral insulin-dependent F4 (per kg)
    a31: float = 1.0e-2
    b33: float = 0.035; c33: float = -32.0
    b32: float = 0.0278; c32: float = -20.2

    # CNS/RBC insulin-independent F5 (per kg)
    a41: float = 1.0e-3; b41: float = 0.031;  c41: float = -50.9
    a42: float = 4.6e-6; b42: float = 0.0144; c42: float = -20.2

    # Glucagon secretion shaping 
    F7_tonic_frac: float = 0.30     
    F7_max_perkg:  float = 100.0    
    F7_floor:      float = 0.0      
    tau7_min:      float = 8.0      

    # Filled by solve_glucagon_gain()
    F7_tonic: float = 0.0           # pg/min (computed)
    F7_max:   float = None          # pg/min (computed)



def set_volumes(p: Params) -> Params:
    mlkg = 1000.0
    p.V1  = 0.20  * p.BW_kg * mlkg
    p.V11 = 0.045 * p.BW_kg * mlkg
    p.V12 = 0.03  * p.BW_kg * mlkg
    p.V13 = 0.10  * p.BW_kg * mlkg
    p.V2  = 0.20  * p.BW_kg * mlkg
    return p

