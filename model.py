import numpy as np
from params import Params
from inputs import Z1_glucose_input, Z2_insulin_input
from processes import (
    scale_bw, F1_LiverProduction_perkg, F2_LiverUptake_perkg, F3_RenalExcretion_perkg,
    F4_PeripheralID_perkg, F5_PeripheralInd_perkg, W_insulin_synthesis, F6_secretion,
    F7_glucagon_secretion
)

def rhs(t, s, P: Params):
    x1,u1p,u2p,u11,u12,u13,u2 = s
    F1 = scale_bw(F1_LiverProduction_perkg(x1,u12,u2,P), P)
    F2 = scale_bw(F2_LiverUptake_perkg(x1,u12,P), P)
    F3 = scale_bw(F3_RenalExcretion_perkg(x1,P), P)
    F4 = scale_bw(F4_PeripheralID_perkg(x1,u13,P), P)
    F5 = scale_bw(F5_PeripheralInd_perkg(x1,P), P)
    NHGB = F1 - F2

    Z1 = Z1_glucose_input(t, P.BW_kg)
    Z2 = Z2_insulin_input(t)

    W  = W_insulin_synthesis(x1,P)
    F6 = F6_secretion(x1,u2p,P)
    F7 = F7_glucagon_secretion(x1,u13,P)

    dx1  = NHGB - F3 - F4 - F5 + Z1
    du1p = -P.k01*u1p + P.k21*u2p + W
    du2p =  P.k01*u1p - P.k12*u2p - F6
    du11 = -(P.m01 + P.m21 + P.m31)*u11 + P.m12*u12 + P.m13*u13 + Z2
    du12 = -(P.m02 + P.m12)*u12 + P.m21*u11 + F6
    du13 = -P.m13*u13 + P.m31*u11
    du2  = -P.h01*u2 + F7
    return np.array([dx1,du1p,du2p,du11,du12,du13,du2], dtype=float)

def integrate(f, t0, tf, dt, y0, P: Params):
    n = int(np.ceil((tf - t0)/dt)) + 1
    T = np.linspace(t0, tf, n)
    Y = np.zeros((n, len(y0)))
    Y[0] = y0
    for i in range(1, n):
        t = T[i-1]; h = T[i]-T[i-1]
        k1 = f(t,         Y[i-1],            P)
        k2 = f(t+0.5*h,   Y[i-1]+0.5*h*k1,   P)
        k3 = f(t+0.5*h,   Y[i-1]+0.5*h*k2,   P)
        k4 = f(t+h,       Y[i-1]+h*k3,       P)
        Y[i] = Y[i-1] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # non-negativity clamp 
        Y[i,1:] = np.maximum(Y[i,1:], 0.0)
    return T, Y
