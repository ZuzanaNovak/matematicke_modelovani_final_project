from params import Params, set_volumes
from processes import solve_glucagon_gain, solve_insulin_basal

def build_control() -> Params:
    P = set_volumes(Params())
    P.a7 = solve_glucagon_gain(P)
    P = solve_insulin_basal(P, u1p0=5e3, u2p0=5e2)
    return P
def build_t2d() -> Params:
    P = set_volumes(Params())
    P.y1_bas = 115.0; P.y2_bas = P.y3_bas = P.y4_bas = 5.0; P.y5_bas = 150.0
    # secretion defect
    P.k01 = 0.02; P.a6 = 1.4; P.c6 = -6.0
    # peripheral IR (F4)
    P.b33 = 0.022; P.c33 = -40.0; P.b32 = 0.022; P.c32 = -10.0; P.a31 = 8.0e-3
    # hepatic IR (F2)
    P.a21 = 0.35e-3; P.a221 = 40.0; P.a222 = 0.6; P.b21 = 0.06; P.c21 = 25.0
    P.b22 = 0.06;    P.c22  = -70.0
    # hepatic production bias (F1)
    P.a11 = 6.2; P.b11 = 1.50; P.c11 = -0.9; P.b13 = 0.022; P.c13 = 24.0
    # glucagon more stubborn
    P.b71 = 0.0015  
    P.c71 = 180.0    
    P.b72 = 0.015    
    P.c72 = 90.0 
    # enforce incomplete suppression
    P.F7_floor = 200.0   

    # insulin kinetics tweak
    P.m01 = 0.125; P.m02 = 0.185; P.m31 = 0.045; P.m13 = 0.022

    # finalize gains
    P.a7 = solve_glucagon_gain(P)
    P = solve_insulin_basal(P, u1p0=5e3, u2p0=5e2)
    return P

