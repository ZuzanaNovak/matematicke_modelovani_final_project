"""
Integrated short-term glucose–insulin–glucagon model (Cobelli et al. style).
Self-contained simulation of a 0.33 g/kg IVGTT (3 min) for a 70 kg subject.

Key fixes vs previous version:
1) Promptly-releasable insulin pool (u2p) now LOSES the secretion term k02(x1)*u2p.
   The secreted insulin appears ONCE in the liver/portal compartment (u12).
2) All glucose unit-process rates (F1..F5) are scaled by body weight (mg/min/kg → mg/min).

Units:
- y1: mg/100 ml (mg/dl)   glucose concentration in plasma/extracellular volume V1
- y2,y3,y4: µU/ml         insulin conc. in plasma (V11), liver/portal (V12), interstitial (V13)
- y5: pg/ml               glucagon concentration in V2
- Time: minutes
- Glucose fluxes F1..F5, NHGB: mg/min (whole subject)
- Insulin secretion/flows: µU/min
- Glucagon secretion/flow: pg/min
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------- numerics ----------
def tanh(z):
    return np.tanh(np.clip(z, -50.0, 50.0))

# ---------- parameters ----------
@dataclass
class Params:
    # Body weight
    BW_kg: float = 70.0

    # Volumes (fractions of BW × 1000 ml/kg)
    V1:  float = None  # glucose: plasma + extracellular
    V11: float = None  # insulin: plasma
    V12: float = None  # insulin: portal/liver
    V13: float = None  # insulin: interstitial
    V2:  float = None  # glucagon

    # Basal concentrations
    y1_bas: float = 91.5   # mg/dl
    y2_bas: float = 11.0   # µU/ml
    y3_bas: float = 11.0   # µU/ml
    y4_bas: float = 11.0   # µU/ml
    y5_bas: float = 75.0   # pg/ml

    # Beta-cell pools (Eqs 2–3)
    k01: float = 0.01      # min^-1 (u1p -> u2p)
    k21: float = 4.34e-3   # min^-1 (u2p -> u1p)
    k12: float = 0.0       # min^-1 (not used explicitly; secretion handled by F6)

    # Insulin synthesis & secretion dependence on glucose (Eqs 26–27)
    a5: float = 0.287      # µU/min (synthesis scale)
    b5: float = 1.51e-2    # (mg/dl)^-1
    c5: float = -92.3      # mg/dl
    a6: float = 1.3        # min^-1  (multiplies u2p)
    b6: float = 9.23e-3    # (mg/dl)^-1
    c6: float = -19.68     # mg/dl

    # Insulin kinetics (Eqs 4–6)
    m01: float = 0.125
    m02: float = 0.185
    m12: float = 0.209
    m13: float = 0.02
    m21: float = 0.26
    m31: float = 0.042

    # Glucagon kinetics (Eq 7)
    h01: float = math.log(2)/8.0  # min^-1

    # Glucagon secretion F7 (Eqs 28–30)
    a7:  float = 2.35             # pg/min
    b71: float = 6.86e-3          # (µU/ml)^-1
    c71: float = 99.2             # µU/ml
    b72: float = 0.086            # (mg/dl)^-1
    c72: float = 40.0             # mg/dl

    # Hepatic production F1 (Eqs 10–13)  [per kg → later scaled by BW]
    a11: float = 1.51             # mg/min/kg
    b11: float = 2.14             # (pg/ml)^-1
    c11: float = -0.85            # pg/ml
    b12: float = 0.784            # (µU/ml)^-1
    c12: float = -108.5           # µU/ml
    b13: float = 0.0275           # (mg/dl)^-1
    c13: float = 20.0             # mg/dl

    # Hepatic uptake F2 (Eqs 14–16)     [per kg]
    a21:  float = 1.95e-3          # mg/min/kg
    b21:  float = 0.0521           # (µU/ml)^-1
    c21:  float = 51.3             # µU/ml
    a221: float = 1.11e2           # dimensionless
    a222: float = 1.45e-2          # dimensionless
    b22:  float = 0.0712           # (mg/dl)^-1
    c22:  float = -108.5           # mg/dl

    # Renal excretion F3 (Eqs 17–19)    [per kg]
    b31:  float = 0.02             # (mg/dl)^-1
    c31:  float = -180.0           # mg/dl
    a321: float = 1.43e-5          # (mg/min/kg)/(mg/dl)
    a322: float = -1.31e-5         # mg/min/kg

    # Peripheral insulin-dependent F4 (Eqs 20–22) [per kg]
    a31:  float = 1.01e-3          # mg/min/kg
    b33:  float = 0.02             # (µU/ml)^-1
    c33:  float = -50.9            # µU/ml
    b32:  float = 0.0278           # (mg/dl)^-1
    c32:  float = -20.2            # mg/dl

    # CNS/RBC insulin-independent F5 (Eqs 23–25) [per kg]
    a41:  float = 2.87e-2          # mg/min/kg
    b41:  float = 0.031            # (mg/dl)^-1
    c41:  float = -50.9            # mg/dl
    a42:  float = 4.6e-6           # mg/min/kg
    b42:  float = 0.0144           # (mg/dl)^-1
    c42:  float = -20.2            # mg/dl

def set_volumes(p: Params) -> Params:
    mlkg = 1000.0
    p.V1  = 0.20  * p.BW_kg * mlkg
    p.V11 = 0.045 * p.BW_kg * mlkg
    p.V12 = 0.03  * p.BW_kg * mlkg
    p.V13 = 0.10  * p.BW_kg * mlkg
    p.V2  = 0.20  * p.BW_kg * mlkg
    return p

P = set_volumes(Params())

# ---------- inputs ----------
def Z1_glucose_input(t: float, P: Params) -> float:
    """IVGTT: 0.33 g/kg over 3 min → mg/min."""
    total_mg = 0.33 * P.BW_kg * 1000.0  # g → mg
    if 0.0 <= t < 3.0:
        return total_mg / 3.0
    return 0.0

def Z2_insulin_input(t: float) -> float:
    """No exogenous insulin."""
    return 0.0

# ---------- helpers ----------
def concentrations(state, P: Params):
    x1,u1p,u2p,u11,u12,u13,u2 = state
    y1 = x1 / P.V1 * 100.0      # mg/dl
    y2 = u11 / P.V11            # µU/ml
    y3 = u12 / P.V12            # µU/ml
    y4 = u13 / P.V13            # µU/ml
    y5 = u2  / P.V2             # pg/ml
    return y1,y2,y3,y4,y5

def deviations(y1,y2,y3,y4,y5,P):
    e1  = y1 - P.y1_bas
    e12 = y3 - P.y3_bas
    e13 = y4 - P.y4_bas
    e21 = y5 - P.y5_bas
    return e1,e12,e13,e21

def scale_bw(rate_mg_min_per_kg: float, P: Params) -> float:
    """Convert mg/min/kg → mg/min (whole subject)."""
    return rate_mg_min_per_kg * P.BW_kg

# ---------- unit processes (return mg/min/kg for glucose; scaled later) ----------
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
    H2 = 0.5*(1 - tanh(P.b21*(e12 + P.c21)))
    M2 = P.a221*(1 + P.a222*0.5*(1 + tanh(P.b22*(e1 + P.c22))))
    return P.a21 * H2 * M2

def F3_RenalExcretion_perkg(x1,P: Params):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    M31 = 0.5*(1 + tanh(P.b31*(y1 + P.c31)))
    M32 = P.a321*y1 + P.a322
    return max(0.0, M31 * M32)

def F4_PeripheralID_perkg(x1,u13,P: Params):
    y1,_,_,y4,_ = concentrations((x1,0,0,0,0,u13,0),P)
    e1,_,e13,_ = deviations(y1,0,0,y4,0,P)
    H3 = 0.5*(1 + tanh(P.b33*(e13 + P.c33)))
    M3 = 0.5*(1 + tanh(P.b32*(e1  + P.c32)))
    return P.a31 * H3 * M3

def F5_PeripheralInd_perkg(x1,P: Params):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    M41 = P.a41 * 0.5*(1 + tanh(P.b41*(e1 + P.c41)))
    M42 = P.a42 * 0.5*(1 + tanh(P.b42*(e1 + P.c42)))
    return M41 + M42

# ---------- insulin & glucagon secretions ----------
def W_insulin_synthesis(x1,P: Params) -> float:
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    return 0.5*P.a5*(1 + tanh(P.b5*(e1 + P.c5)))  # µU/min

def F6_secretion(x1,u2p,P: Params) -> float:
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    return 0.5*P.a6*(1 + tanh(P.b6*(e1 + P.c6))) * u2p  # µU/min

def F7_glucagon_secretion(x1,u13,P: Params) -> float:
    y1,_,_,y4,_ = concentrations((x1,0,0,0,0,u13,0),P)
    e1,_,e13,_ = deviations(y1,0,0,y4,0,P)
    H7 = 0.5*(1 - tanh(P.b71*(e13 + P.c71)))
    M7 = 0.5*(1 - tanh(P.b72*(e1  + P.c72)))
    return P.a7 * H7 * M7  # pg/min

# ---------- ODE system ----------
def rhs(t, state, P: Params):
    x1,u1p,u2p,u11,u12,u13,u2 = state

    # glucose unit processes per kg → scale to whole subject mg/min
    F1 = scale_bw(F1_LiverProduction_perkg(x1,u12,u2,P), P)
    F2 = scale_bw(F2_LiverUptake_perkg(x1,u12,P), P)
    F3 = scale_bw(F3_RenalExcretion_perkg(x1,P), P)
    F4 = scale_bw(F4_PeripheralID_perkg(x1,u13,P), P)
    F5 = scale_bw(F5_PeripheralInd_perkg(x1,P), P)

    NHGB = F1 - F2
    Z1 = Z1_glucose_input(t,P)
    Z2 = Z2_insulin_input(t)

    # insulin/glucagon secretions
    W  = W_insulin_synthesis(x1,P)         # µU/min
    F6 = F6_secretion(x1,u2p,P)            # µU/min
    F7 = F7_glucagon_secretion(x1,u13,P)   # pg/min

    # ODEs
    dx1  = NHGB - F3 - F4 - F5 + Z1                     # Eq (1)
    du1p = -P.k01*u1p + P.k21*u2p + W                   # Eq (2)
    du2p =  P.k01*u1p - (P.k12)*u2p - F6                # Eq (3)  (FIXED)
    du11 = -(P.m01 + P.m21 + P.m31)*u11 + P.m12*u12 + P.m13*u13 + Z2   # Eq (4)
    du12 = -(P.m02 + P.m12)*u12 + P.m21*u11 + F6                        # Eq (5) (+F6 ONCE)
    du13 = -P.m13*u13 + P.m31*u11                                      # Eq (6)
    du2  = -P.h01*u2 + F7                                              # Eq (7)

    return np.array([dx1,du1p,du2p,du11,du12,du13,du2], dtype=float)

# ---------- initial state ----------
def initial_state(P: Params):
    # Glucose mass from basal concentration
    x1_0 = (P.y1_bas/100.0) * P.V1  # mg

    # Basal insulin/glucagon amounts from concentrations
    u11_0 = P.y2_bas * P.V11  # µU
    u12_0 = P.y3_bas * P.V12  # µU
    u13_0 = P.y4_bas * P.V13  # µU
    u2_0  = P.y5_bas * P.V2   # pg

    # Pancreatic stores (paper-scale units → µU)
    u1p_0 = 4.9e6   # µU (stored)
    u2p_0 = 4.9e5   # µU (prompt)

    return np.array([x1_0,u1p_0,u2p_0,u11_0,u12_0,u13_0,u2_0], dtype=float)

# ---------- integrator (RK4) ----------
def integrate(f, t0, tf, dt, y0, P):
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
    return T, Y

# ---------- run ----------
os.makedirs("results", exist_ok=True)

P = set_volumes(P)
y0 = initial_state(P)
t0, tf, dt = 0.0, 180.0, 0.05
T, Y = integrate(rhs, t0, tf, dt, y0, P)

# ---------- derived series ----------
x1,u1p,u2p,u11,u12,u13,u2 = Y.T
y1 = x1 / P.V1 * 100.0
y2 = u11 / P.V11
y3 = u12 / P.V12
y4 = u13 / P.V13
y5 = u2  / P.V2

# recompute fluxes for plotting
F1 = np.array([scale_bw(F1_LiverProduction_perkg(x,u12_i,u2_i,P), P) for x,u12_i,u2_i in zip(x1,u12,u2)])
F2 = np.array([scale_bw(F2_LiverUptake_perkg(x,u12_i,P), P)          for x,u12_i in zip(x1,u12)])
F3 = np.array([scale_bw(F3_RenalExcretion_perkg(x,P), P)             for x in x1])
F4 = np.array([scale_bw(F4_PeripheralID_perkg(x,u13_i,P), P)         for x,u13_i in zip(x1,u13)])
F5 = np.array([scale_bw(F5_PeripheralInd_perkg(x,P), P)              for x in x1])
NHGB = F1 - F2

# ---------- plotting ----------
# 1) Glucose
plt.figure()
plt.plot(T, y1)
plt.xlabel("Time (min)")
plt.ylabel("Plasma glucose y1 (mg/100 ml)")
plt.title("Plasma Glucose (IVGTT 0.33 g/kg over 3 min)")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/01_glucose.png", dpi=200); plt.close()

# 2) Insulin concentrations (µU/ml) – same units stacked
fig, ax = plt.subplots(3,1, figsize=(7,8), sharex=True)
ax[0].plot(T, y2); ax[0].set_ylabel("Plasma y2 (µU/ml)")
ax[1].plot(T, y3); ax[1].set_ylabel("Portal/Liver y3 (µU/ml)")
ax[2].plot(T, y4); ax[2].set_ylabel("Interstitial y4 (µU/ml)")
ax[2].set_xlabel("Time (min)")
for a in ax: a.grid(True, alpha=0.3)
fig.suptitle("Insulin Concentrations")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("results/02_insulin_concentrations.png", dpi=200); plt.close(fig)

# 3) Glucagon (pg/ml)
plt.figure()
plt.plot(T, y5)
plt.xlabel("Time (min)"); plt.ylabel("Glucagon y5 (pg/ml)")
plt.title("Glucagon Concentration")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/03_glucagon.png", dpi=200); plt.close()

# 4) Hepatic fluxes (same units mg/min)
plt.figure()
plt.plot(T, F1, label="F1: Hepatic production")
plt.plot(T, F2, label="F2: Hepatic uptake")
plt.plot(T, NHGB, label="NHGB = F1 - F2")
plt.xlabel("Time (min)"); plt.ylabel("Rate (mg/min)")
plt.title("Hepatic Glucose Processes")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/04_hepatic_processes.png", dpi=200); plt.close()

# 5) Peripheral & renal (same units mg/min)
plt.figure()
plt.plot(T, F4, label="F4: Peripheral insulin-dependent")
plt.plot(T, F5, label="F5: CNS/RBC insulin-independent")
plt.plot(T, F3, label="F3: Renal excretion")
plt.xlabel("Time (min)"); plt.ylabel("Rate (mg/min)")
plt.title("Peripheral Utilizations & Renal Excretion")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/05_peripheral_and_renal.png", dpi=200); plt.close()

# 6) Pancreatic pools (same units µU)
plt.figure()
plt.plot(T, u1p, label="Stored u1p (µU)")
plt.plot(T, u2p, label="Prompt u2p (µU)")
plt.xlabel("Time (min)"); plt.ylabel("Amount (µU)")
plt.title("Pancreatic Insulin Pools")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/06_pancreatic_pools.png", dpi=200); plt.close()

# 7) Plasma vs interstitial insulin (same units)
plt.figure()
plt.plot(T, y2, label="Plasma y2")
plt.plot(T, y4, label="Interstitial y4")
plt.xlabel("Time (min)"); plt.ylabel("Insulin (µU/ml)")
plt.title("Plasma vs Interstitial Insulin")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/07_insulin_plasma_vs_interstitial.png", dpi=200); plt.close()

print("Done. Figures saved to ./results/")
