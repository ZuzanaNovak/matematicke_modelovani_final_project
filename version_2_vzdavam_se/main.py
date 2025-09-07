"""
Cobelli-style short-term glucose–insulin–glucagon model
IVGTT 0.33 g/kg over 3 min, 70 kg — paperfit_v3

Goals vs v2:
- Hepatic NHGB shows a proper "U": strong negative early, then recovery toward ~0.
- No late F1 surge or glucose bump.
- Cohort-like basals (77 mg/dl glucose; 7 µU/ml insulin; 120 pg/ml glucagon).

Key changes:
- F1 (hepatic production): a11=5.5, b11=1.60, c11=-1.10, c13=26  → softer glucagon drive & gentler recovery.
- F2 (hepatic uptake): a21=0.5e-3, b21=0.10, c21=10.0, a221=60, a222=1.00, b22=0.090, c22=-90
  → strong early uptake that relaxes with falling insulin/glucose.
- Glucagon inhibition: b71=0.0053, c71=115, b72=0.070, c72=52; a7 solved from steady state
  → sharp fall, moderate rebound without overshoot.
- F4 magnitude trimmed to 5.5e2 (from 6e2).
"""

import os, math, numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def tanh(z): return np.tanh(np.clip(z, -50.0, 50.0))

@dataclass
class Params:
    BW_kg: float = 70.0
    V1:  float = None; V11: float = None; V12: float = None; V13: float = None; V2: float = None

    # Cohort basals
    y1_bas: float = 77.0  # mg/dl
    y2_bas: float = 7.0   # µU/ml
    y3_bas: float = 7.0   # µU/ml
    y4_bas: float = 7.0   # µU/ml
    y5_bas: float = 120.0 # pg/ml

    # Beta-cell pools / secretion
    k01: float = 0.03; k21: float = 4.34e-3; k12: float = 0.0
    a5: float = 0.287; b5: float = 1.51e-2; c5: float = -92.3
    a6: float = 2.4;   b6: float = 9.23e-3; c6: float = -10.0

    # Insulin kinetics (slightly slower tail; earlier y4)
    m01: float = 0.110
    m02: float = 0.170
    m12: float = 0.209
    m13: float = 0.018   # slower interstitial clearance
    m21: float = 0.26
    m31: float = 0.060   # faster plasma→interstitial

    # Glucagon kinetics
    h01: float = math.log(2)/8.0

    # Glucagon secretion (softened rebound); a7 solved at runtime
    a7:  float = None
    b71: float = 0.0053   # insulin inhibition slope ↑
    c71: float = 115.0    # shift right ↑
    b72: float = 0.070    # glucose inhibition slope
    c72: float = 52.0     # shift right ↑

    # Hepatic production F1 (per kg)
    a11: float = 5.5      # ↓ cap on max production
    b11: float = 1.60     # ↓ glucagon sensitivity
    c11: float = -1.10    # slight bias
    b12: float = 0.784; c12: float = -108.5
    b13: float = 0.0275
    c13: float = 26.0     # ↓ vs v2 to avoid late surge

    # Hepatic uptake F2 (per kg) — make it relax later
    a21:  float = 0.5e-3
    b21:  float = 0.10
    c21:  float = 10.0
    a221: float = 60.0
    a222: float = 1.00    # much more glucose effect (0→1 factor)
    b22:  float = 0.090
    c22:  float = -90.0

    # Renal excretion F3 (per kg)
    b31:  float = 0.02; c31: float = -190.0
    a321: float = 1.43e-5; a322: float = -1.31e-5

    # Peripheral uptake F4 (per kg)
    a31:  float = 1.0e-2
    b33:  float = 0.035
    c33:  float = -32.0
    b32:  float = 0.0278; c32:  float = -20.2

    # CNS/RBC uptake F5 (per kg)
    a41:  float = 1.0e-3; b41: float = 0.031; c41: float = -50.9
    a42:  float = 4.6e-6; b42: float = 0.0144; c42: float = -20.2

def set_volumes(p: Params):
    mlkg = 1000.0
    p.V1  = 0.20  * p.BW_kg * mlkg
    p.V11 = 0.045 * p.BW_kg * mlkg
    p.V12 = 0.03  * p.BW_kg * mlkg
    p.V13 = 0.10  * p.BW_kg * mlkg
    p.V2  = 0.20  * p.BW_kg * mlkg
    return p

P = set_volumes(Params())

def Z1_glucose_input(t, P):
    total_mg = 0.33 * P.BW_kg * 1000.0
    return total_mg/3.0 if (0.0 <= t < 3.0) else 0.0

def Z2_insulin_input(t): return 0.0

def concentrations(state, P):
    x1,u1p,u2p,u11,u12,u13,u2 = state
    return (x1/P.V1*100.0, u11/P.V11, u12/P.V12, u13/P.V13, u2/P.V2)

def deviations(y1,y2,y3,y4,y5,P):
    return (y1-P.y1_bas, y3-P.y3_bas, y4-P.y4_bas, y5-P.y5_bas)

def scale_bw(rate_perkg, P): return rate_perkg * P.BW_kg

# ---------- Unit processes (per kg) ----------
def F1_LiverProduction_perkg(x1,u12,u2,P):
    y1,_,y3,_,y5 = concentrations((x1,0,0,0,u12,0,u2),P)
    e1,e12,_,e21 = deviations(y1,0,y3,0,y5,P)
    G1 = 0.5*(1 + tanh(P.b11*(e21 + P.c11)))
    H1 = 0.5*(1 - tanh(P.b12*(e12 + P.c12)))
    M1 = 0.5*(1 - tanh(P.b13*(e1  + P.c13)))
    return P.a11 * G1 * H1 * M1

def F2_LiverUptake_perkg(x1,u12,P):
    y1,_,y3,_,_ = concentrations((x1,0,0,0,u12,0,0),P)
    e1,e12,_,_ = deviations(y1,0,y3,0,0,P)
    H2 = 0.5*(1 + tanh(P.b21*(e12 + P.c21)))               # insulin control
    M2 = P.a221 * (1 + P.a222 * 0.5*(1 + tanh(P.b22*(e1 + P.c22))))  # glucose control
    return P.a21 * H2 * M2

def F3_RenalExcretion_perkg(x1,P):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    M31 = 0.5*(1 + tanh(P.b31*(y1 + P.c31)))
    M32 = P.a321*y1 + P.a322
    return max(0.0, M31*M32)

def F4_PeripheralID_perkg(x1,u13,P):
    y1,_,_,y4,_ = concentrations((x1,0,0,0,0,u13,0),P)
    e1,_,e13,_ = deviations(y1,0,0,y4,0,P)
    H3 = 0.5*(1 + tanh(P.b33*(e13 + P.c33)))
    M3 = 0.5*(1 + tanh(P.b32*(e1  + P.c32)))
    return 5.5e2 * (P.a31 * H3 * M3)   # trimmed magnitude

def F5_PeripheralInd_perkg(x1,P):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    M41 = P.a41 * 0.5*(1 + tanh(P.b41*(e1 + P.c41)))
    M42 = P.a42 * 0.5*(1 + tanh(P.b42*(e1 + P.c42)))
    return 3e2 * (M41 + M42)

# ---------- Secretions ----------
def W_insulin_synthesis(x1,P):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    return 0.5*P.a5*(1 + tanh(P.b5*(e1 + P.c5)))

def F6_secretion(x1,u2p,P):
    y1,_,_,_,_ = concentrations((x1,0,0,0,0,0,0),P)
    e1,_,_,_ = deviations(y1,0,0,0,0,P)
    return 0.5*P.a6*(1 + tanh(P.b6*(e1 + P.c6))) * u2p

def F7_glucagon_secretion(x1,u13,P):
    y1,_,_,y4,_ = concentrations((x1,0,0,0,0,u13,0),P)
    e1,_,e13,_ = deviations(y1,0,0,y4,0,P)
    H7 = 0.5*(1 - tanh(P.b71*(e13 + P.c71)))  # insulin inhibition
    M7 = 0.5*(1 - tanh(P.b72*(e1  + P.c72)))  # glucose inhibition
    return P.a7 * H7 * M7

def solve_glucagon_gain(P):
    H7b = 0.5*(1 - np.tanh(P.b71*(0 + P.c71)))
    M7b = 0.5*(1 - np.tanh(P.b72*(0 + P.c72)))
    target = P.h01 * P.y5_bas * P.V2
    return target / max(1e-12, H7b*M7b)
def solve_insulin_basal(P, u1p0, u2p0):
    """
    Choose c6 (secretion setpoint) so F6_bas = k01*u1p0
    and a5 (synthesis scale) so W_bas = k01*u1p0 - k21*u2p0.
    Keeps u1p/u2p steady at basal.
    """
    # --- match secretion at basal: F6 = 0.5*a6*(1+tanh(b6*c6))*u2p0 = k01*u1p0
    q = (2.0*P.k01*u1p0)/(P.a6*u2p0) - 1.0               # = tanh(b6*c6)
    q = float(np.clip(q, -0.999, 0.999))
    P.c6 = (np.arctanh(q))/P.b6

    # --- match synthesis at basal: W = 0.5*a5*(1+tanh(b5*c5)) = k01*u1p0 - k21*u2p0
    W0 = P.k01*u1p0 - P.k21*u2p0
    g  = 0.5*(1.0 + np.tanh(P.b5*(0.0 + P.c5)))          # basal activation
    P.a5 = W0 / max(g, 1e-12)
    return P


# ---------- ODEs ----------
def rhs(t, s, P):
    x1,u1p,u2p,u11,u12,u13,u2 = s
    F1 = scale_bw(F1_LiverProduction_perkg(x1,u12,u2,P), P)
    F2 = scale_bw(F2_LiverUptake_perkg(x1,u12,P), P)
    F3 = scale_bw(F3_RenalExcretion_perkg(x1,P), P)
    F4 = scale_bw(F4_PeripheralID_perkg(x1,u13,P), P)
    F5 = scale_bw(F5_PeripheralInd_perkg(x1,P), P)
    NHGB = F1 - F2
    Z1 = Z1_glucose_input(t,P); Z2 = Z2_insulin_input(t)

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
    return np.array([dx1,du1p,du2p,du11,du12,du13,du2], float)

# ---------- run ----------
def initial_state(P):
    x1_0 = (P.y1_bas/100.0)*P.V1
    u11_0 = P.y2_bas*P.V11; u12_0 = P.y3_bas*P.V12; u13_0 = P.y4_bas*P.V13
    u2_0  = P.y5_bas*P.V2
    u1p_0 = 4.9e6; u2p_0 = 4.9e5
    return np.array([x1_0,u1p_0,u2p_0,u11_0,u12_0,u13_0,u2_0], float)

def integrate(f, t0, tf, dt, y0, P):
    n = int(np.ceil((tf - t0)/dt)) + 1
    T = np.linspace(t0, tf, n); Y = np.zeros((n, len(y0))); Y[0] = y0
    for i in range(1, n):
        t = T[i-1]; h = T[i]-T[i-1]
        k1 = f(t,       Y[i-1],          P)
        k2 = f(t+h/2.0, Y[i-1]+h*k1/2.0, P)
        k3 = f(t+h/2.0, Y[i-1]+h*k2/2.0, P)
        k4 = f(t+h,     Y[i-1]+h*k3,     P)
        Y[i] = Y[i-1] + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return T, Y

os.makedirs("results_paperfit_v3", exist_ok=True)
P = set_volumes(P); P.a7 = solve_glucagon_gain(P)
P = solve_insulin_basal(P, u1p0=4.9e6, u2p0=4.9e5)
T, Y = integrate(rhs, 0.0, 180.0, 0.05, initial_state(P), P)

# ---------- derived & plots ----------
x1,u1p,u2p,u11,u12,u13,u2 = Y.T
y1 = x1/P.V1*100.0; y2 = u11/P.V11; y3 = u12/P.V12; y4 = u13/P.V13; y5 = u2/P.V2

F1 = np.array([scale_bw(F1_LiverProduction_perkg(x,uu12,uu2,P), P) for x,uu12,uu2 in zip(x1,u12,u2)])
F2 = np.array([scale_bw(F2_LiverUptake_perkg(x,uu12,P), P)        for x,uu12 in zip(x1,u12)])
F3 = np.array([scale_bw(F3_RenalExcretion_perkg(x,P), P)          for x in x1])
F4 = np.array([scale_bw(F4_PeripheralID_perkg(x,uu13,P), P)       for x,uu13 in zip(x1,u13)])
F5 = np.array([scale_bw(F5_PeripheralInd_perkg(x,P), P)           for x in x1])
NHGB = F1 - F2

def v_at(tmin, T, s): return float(s[np.argmin(np.abs(T-tmin))])

plt.figure(); plt.plot(T,y1); plt.xlabel("Time (min)"); plt.ylabel("Glucose (mg/dl)")
plt.title("Plasma Glucose — paperfit_v3"); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig("results_paperfit_v3/01_glucose.png", dpi=200); plt.close()

fig,ax = plt.subplots(3,1,figsize=(7,8),sharex=True)
ax[0].plot(T,y2); ax[0].set_ylabel("Plasma (µU/ml)")
ax[1].plot(T,y3); ax[1].set_ylabel("Portal/Liver (µU/ml)")
ax[2].plot(T,y4); ax[2].set_ylabel("Interstitial (µU/ml)"); ax[2].set_xlabel("Time (min)")
for a in ax: a.grid(True,alpha=0.3)
fig.suptitle("Insulin Concentrations — paperfit_v3")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("results_paperfit_v3/02_insulin_concs.png", dpi=200); plt.close(fig)

plt.figure(); plt.plot(T,y5); plt.xlabel("Time (min)"); plt.ylabel("Glucagon (pg/ml)")
plt.title("Glucagon — paperfit_v3"); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig("results_paperfit_v3/03_glucagon.png", dpi=200); plt.close()

plt.figure()
plt.plot(T,F1,label="F1 (hepatic prod)")
plt.plot(T,F2,label="F2 (hepatic uptake)")
plt.plot(T,NHGB,label="NHGB = F1 - F2")
plt.xlabel("Time (min)"); plt.ylabel("mg/min"); plt.title("Hepatic Processes — paperfit_v3")
plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig("results_paperfit_v3/04_hepatic.png", dpi=200); plt.close()

plt.figure()
plt.plot(T,F4,label="F4 peripheral (ID)")
plt.plot(T,F5,label="F5 CNS/RBC (IND)")
plt.plot(T,F3,label="F3 renal")
plt.xlabel("Time (min)"); plt.ylabel("mg/min")
plt.title("Peripheral & Renal — paperfit_v3")
plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig("results_paperfit_v3/05_periph_renal.png", dpi=200); plt.close()

plt.figure()
plt.plot(T,u1p,label="u1p stored (µU)")
plt.plot(T,u2p,label="u2p prompt (µU)")
plt.xlabel("Time (min)"); plt.ylabel("µU")
plt.title("Pancreatic Pools — paperfit_v3")
plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig("results_paperfit_v3/06_pools.png", dpi=200); plt.close()

plt.figure()
plt.plot(T,y2,label="Plasma")
plt.plot(T,y4,label="Interstitial")
plt.xlabel("Time (min)"); plt.ylabel("Insulin (µU/ml)")
plt.title("Plasma vs Interstitial — paperfit_v3")
plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig("results_paperfit_v3/07_y2_vs_y4.png", dpi=200); plt.close()

print("== Summary (paperfit_v3) ==")
print(f"Basal glucose: {P.y1_bas:.1f} mg/dl")
print(f"Glucose peak: {float(np.max(y1)):.1f} mg/dl | @60 min: {v_at(60,T,y1):.1f} | @120 min: {v_at(120,T,y1):.1f}")
print(f"Plasma insulin peak: {float(np.max(y2)):.1f} µU/ml | @120 min: {v_at(120,T,y2):.1f} µU/ml")
print(f"Glucagon nadir: {float(np.min(y5)):.1f} pg/ml | end: {v_at(180,T,y5):.1f} pg/ml")
print(f"NHGB min: {float(np.min(NHGB)):.2f} mg/min at t≈{T[np.argmin(NHGB)]:.1f} min")
print(f"NHGB @60: {v_at(60,T,NHGB):.2f} mg/min | @120: {v_at(120,T,NHGB):.2f} mg/min")
print("Figures saved to ./results_paperfit_v3")
