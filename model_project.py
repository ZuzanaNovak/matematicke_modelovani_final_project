"""
Reimplementation of Chew et al. (2009) "Modeling of oscillatory bursting activity of pancreatic beta-cells under regulated glucose stimulation"
- Dual Oscillator Model (glycolysis + mitochondria + electrical/ER Ca2+)
- Cobelli & Mari whole-body glucose regulation (glucose–insulin–glucagon)
- Coupling via plasma glucose -> GLUT2 uptake -> intracellular glucose -> J_GK

Notes
-----
• Units follow the paper (mixed units). Values are taken as-is for reproducibility.
• We integrate in **milliseconds** for numerical stability. Cobelli (per minute) is scaled internally by /60000.
• Demo simulates a 70 kg subject with an IV glucose load of 100 mg/kg/min for 3 minutes and runs 120 minutes total.

Author: ChatGPT
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

try:
    from scipy.integrate import solve_ivp
except Exception as e:
    raise RuntimeError("This script requires SciPy. Please `pip install scipy`.\n" + str(e))

# ---------- utilities ----------

def safe_exp(z, clip=50.0):
    """Exponent with clipping to avoid overflow/NaNs."""
    return np.exp(np.clip(z, -clip, clip))

# ---------------------------
# Parameters (Appendix A & B)
# ---------------------------

@dataclass
class DualOscParams:
    # Glycolytic component (Table A1)
    Vglut: float = 8.0          # mM/ms
    Kglut: float = 7.0          # mM
    Vgk: float = 0.8            # mM/ms
    Kgk: float = 7.0            # mM
    ngk: float = 4.0
    kGPDH: float = 0.0005       # µM/ms
    Vmax: float = 0.005         # µM/ms
    phi: float = 0.06           # dimensionless
    K1: float = 30.0            # µM
    K2: float = 1.0             # µM
    K3: float = 50000.0         # µM
    K4: float = 220.0           # µM
    f13: float = 0.02
    f23: float = 0.2
    f41: float = 20.0
    f42: float = 20.0
    f43: float = 20.0

    # Mitochondrial component (Table A1)
    p1: float = 400.0
    p2: float = 1.0
    p3: float = 0.01            # µM
    p4: float = 0.6             # µM/ms
    p5: float = 0.1             # mM
    p6: float = 177.0           # mV
    p7: float = 5.0             # mV
    p8: float = 7.0             # µM/ms
    p9: float = 0.1             # mM
    p10: float = 177.0          # mV
    p11: float = 5.0            # mV
    p13: float = 10.0           # mM
    p14: float = 190.0          # mV
    p15: float = 8.5            # mV
    p16: float = 35.0           # µM/ms
    p17: float = 0.002          # µM ms^-1 mV^-1
    p18: float = -0.03          # µM/ms
    p19: float = 0.35           # µM/ms
    p20: float = 2.0
    p21: float = 0.04           # µM ms^-1 mV^-1
    p22: float = 1.1            # µM ms^-1
    p23: float = 0.01           # ms^-1
    p24: float = 0.016          # mV^-1
    JGPDH_bas: float = 0.0005   # µM/ms
    fm: float = 0.01
    NADm_tot: float = 10.0      # mM
    Am_tot: float = 15.0        # mM
    Cm: float = 1.8             # µM/mV

    # Electrical / Ca2+ (Table A1)
    C: float = 5300.0           # fF
    tau_n: float = 20.0         # ms
    gK: float = 2700.0          # pS
    gCa: float = 1000.0         # pS
    gK_Ca: float = 300.0        # pS
    gK_ATP: float = 16000.0     # pS
    VK: float = -75.0           # mV
    VCa: float = 25.0           # mV

    D: float = 0.5              # µM
    khyd: float = 5e-5          # ms^-1 µM^-1
    khyd_bas: float = 5e-5      # ms^-1
    alpha: float = 4.5e-6       # µM/ms per pA
    kPMCA: float = 0.1          # ms^-1

    fc: float = 0.01
    fer: float = 0.01
    Vc_over_Ver: float = 31.0
    Ca_bas: float = 0.05        # µM

    pleak: float = 0.0002       # ms^-1
    kSERCA: float = 0.4         # ms^-1

    # Cytosolic nucleotide pool (A.37–A.39)
    Ac_tot: float = 2500.0      # µM
    AMP_c: float = 500.0        # µM (fixed)

    # Geometry ratio
    vol_ratio: float = 0.07     # mitochondria/cytosol volume ratio


@dataclass
class CobelliParams:
    # Distribution volumes (fractions of body weight)
    BW_kg: float = 70.0
    V1_frac: float = 0.20
    V11_frac: float = 0.045
    V12_frac: float = 0.031
    V13_frac: float = 0.106
    V2_frac: float = 0.20

    # Basal concentrations
    y1_basal: float = 91.5      # mg/100 ml (plasma glucose)
    y2_basal: float = 11.0      # µU/ml (plasma insulin)
    y5_basal: float = 75.0      # pg/ml (blood glucagon)

    # Glucose subsystem
    a11: float = 6.727; b11: float = 2.1451; c11: float = -0.854
    b12: float = 1.18e-2; c12: float = -100.0
    b13: float = 0.15; c13: float = 9.8
    b14: float = 5.9e-2; c14: float = -20.0

    a21: float = 0.56
    b21: float = 1.54e-2; c21: float = -172.0
    a22: float = 0.105
    b22: float = 0.1; c22: float = -25.0

    a31: float = 7.14e-6
    a32: float = -9.15e-2
    b31: float = 0.143
    c31: float = -2.52e4

    a41: float = 3.45
    b41: float = 1.4e-2; c41: float = -146.0
    b42: float = 1.4e-2; c42: float = 0.0

    a51: float = 8.4e-2
    a52: float = 3.22e-4
    a53: float = 3.29e-2
    b51: float = 2.78e-2; c51: float = 91.5

    # Insulin subsystem
    k12: float = 3.128e-2
    k21: float = 4.34e-3
    aw: float = 0.74676
    a6: float = 1.0
    bw: float = 1.09e-2; cw: float = -175.0
    b6: float = 5e-2; c6: float = -44.0

    m01: float = 0.125
    m02: float = 0.185
    m12: float = 0.209
    m13: float = 0.02
    m21: float = 0.268
    m31: float = 0.042

    # Glucagon subsystem
    a71: float = 2.3479
    b71: float = 6.86e-4; c71: float = 99.246
    b72: float = 3e-2; c72: float = 40.0
    h02: float = 0.086643

    # Volumes in "100 ml" units (mg/100 ml convention)
    @property
    def V1(self):  return self.V1_frac  * self.BW_kg * 1e3
    @property
    def V11(self): return self.V11_frac * self.BW_kg * 1e3
    @property
    def V12(self): return self.V12_frac * self.BW_kg * 1e3
    @property
    def V13(self): return self.V13_frac * self.BW_kg * 1e3
    @property
    def V2(self):  return self.V2_frac  * self.BW_kg * 1e3

# ---------------------------
# Helper nonlinear functions (Cobelli)
# ---------------------------

def tanh(x): return np.tanh(x)

def cobelli_functions(u, p: CobelliParams):
    u1, u1p, u2p, u11, u12, u13, u2 = u
    Ge = u1 / p.V1         # mg/100 ml
    y2 = u11 / p.V11       # µU/ml
    y3 = u12 / p.V12
    y4 = u13 / p.V13
    y5 = u2 / p.V2         # pg/ml

    ex  = Ge - p.y1_basal
    e12 = y3 - p.y2_basal
    e21 = y5 - p.y5_basal
    e13 = y4 - p.y2_basal

    # Liver production F1
    G1 = 0.5 * (1 + tanh(p.b11 * (e21 + p.c11)))
    H1 = 0.5 * (1 - tanh(p.b12 * (e12 + p.c12)) + (1 - tanh(p.b13 * (e12 + p.c13))))
    M1 = 0.5 * (1 - tanh(p.b14 * (ex  + p.c14)))
    F1 = p.a11 * G1 * H1 * M1

    # Liver uptake F2
    H2 = 0.5 * p.a21 * (1 + tanh(p.b21 * (e12 + p.c21)))
    M2 = 0.5 * p.a22 * (1 + tanh(p.b22 * (ex  + p.c22)))
    F2 = H2 + M2

    # Renal excretion F3
    if u1 > 2.52e4:
        M31 = 0.5 * (1 + tanh(p.b31 * (u1 + p.c31)))
        M32 = p.a31 * u1 + p.a32
        F3 = M31 * M32
    else:
        F3 = 0.0

    # Peripheral utilization
    H4 = 0.5 * (1 + tanh(p.b41 * (e13 + p.c41)))
    M4 = 0.5 * (1 + tanh(p.b42 * (ex  + p.c42)))
    F4 = p.a41 * H4 * M4

    # Insulin-independent utilization
    M51 = p.a51 * tanh(p.b51 * (ex + p.c51))
    M52 = p.a52 * ex + p.a53
    F5 = M51 + M52

    # Insulin synthesis and secretion
    W  = 0.5 * p.aw * (1 + tanh(p.bw * (ex + p.cw)))
    F6 = 0.5 * p.a6 * (1 + tanh(p.b6 * (ex + p.c6))) * u2p

    # Glucagon secretion
    H7 = 0.5 * (1 - tanh(p.b71 * (e13 + p.c71)))
    M7 = 0.5 * (1 - tanh(p.b72 * (ex  + p.c72)))
    F7 = p.a71 * H7 * M7

    return Ge, y2, y3, y4, y5, ex, F1, F2, F3, F4, F5, W, F6, F7

# ---------------------------
# Dual Oscillator pieces
# ---------------------------

def pfk_rate(F6P_uM, FBP_uM, ATP_uM, AMP_uM, par: DualOscParams):
    # Equations (A.6–A.7)
    def omega(i, j, k, l):
        f13, f23, f41, f42, f43 = par.f13, par.f23, par.f41, par.f42, par.f43
        K1, K2, K3, K4 = par.K1, par.K2, par.K3, par.K4
        return (1.0/((f13**i)*(f23**j)*(f41**i)*(f42**j)*(f43**k))) * \
               ((AMP_uM/K1)**i) * ((FBP_uM/K2)**j) * ((F6P_uM/K3)**k) * ((ATP_uM/K4)**l)

    den = 0.0
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                for l in (0,1):
                    den += omega(i,j,k,l)
    num = (1 - par.phi)*omega(1,1,1,0) + par.phi*sum(omega(i,j,1,l) for i in (0,1) for j in (0,1) for l in (0,1))
    return par.Vmax * (num/den)  # µM/ms

def dual_oscillator_rhs(t_min, x, Ge_mg_per_100ml: float, par: DualOscParams):
    """
    x:
      [Gi (mM), G6P (µM), FBP (µM), NADHm (mM), phi_m (mV),
       Ca_m (µM), ADPm (mM), V (mV), n, Ca_c (µM), Ca_er (µM), ATP_c (µM)]
    """
    Gi, G6P, FBP, NADHm, phi_m, Ca_m, ADPm, V, n, Ca_c, Ca_er, ATP_c = x

    # GLUT2 transport + GK
    Ge_mM = Ge_mg_per_100ml/18.0  # use as-is (paper's mixed units)
    Jglut = par.Vglut * (Ge_mM - Gi) * par.Kglut / ((par.Kglut + Ge_mM)*(par.Kglut + Gi))  # mM/ms
    JGK   = par.Vgk * (Gi**par.ngk) / (par.Kgk**par.ngk + Gi**par.ngk)                     # mM/ms
    dGi   = Jglut - JGK

    # Glycolysis
    F6P   = 0.3 * G6P
    AMP_uM, ATP_uM = par.AMP_c, ATP_c
    JPFK  = pfk_rate(F6P, FBP, ATP_uM, AMP_uM, par)
    JGPDH = par.kGPDH * np.sqrt(max(FBP, 0.0))  # µM/ms
    dG6P  = (JGK*1000.0) - JPFK                 # mM->µM
    dFBP  = JPFK - 0.5*JGPDH

    # Mitochondria
    phi_eff = np.clip(phi_m, 120.0, 200.0) 
    NADm = max(par.NADm_tot - NADHm, 0.0)
    RATm = max((par.Am_tot - ADPm)/max(ADPm, 1e-12), 0.0)  # ATPm/ADPm
    JPDH = ((par.p1 * NADm) / (par.p2 * NADm + NADHm)) * (Ca_m / (par.p3 + Ca_m)) * (JGPDH + par.JGPDH_bas)
    Jo   = ((par.p4 * NADHm) / (par.p5 + NADHm)) * (1.0 / (1.0 + safe_exp((phi_eff - par.p6)/par.p7)))
    dNADHm = 0.001 * (JPDH - Jo)

    JH_res = ((par.p8 * NADHm) / (par.p9 + NADHm)) * (1.0 / (1.0 + safe_exp((phi_m - par.p10)/par.p11)))
    JF1F0  = (par.p13 / (par.p13 + (par.Am_tot - ADPm))) * (par.p16 / (1.0 + safe_exp((par.p14 - phi_eff)/par.p15)))
    JH_atp = 3.0 * JF1F0
    JANT   = par.p19 * (RATm / (RATm + par.p20)) * safe_exp(0.5 * 0.037 * phi_eff)
    JH_leak = par.p17 * phi_m + par.p18
    Juni   = (par.p21 * phi_m - par.p22) * Ca_c
    JNaCa  = par.p23 * (Ca_m / max(Ca_c, 1e-9)) * safe_exp(par.p24 * phi_eff)

    dphi_m = (JH_res - JH_atp - JANT - JH_leak - JNaCa - 2.0*Juni) / par.Cm
    

    dCa_m  = -par.fm * (JNaCa - Juni)
    dADPm  = 0.001 * (JANT - JF1F0)

    # Electrical & Ca2+
    n_inf = 1.0 / (1.0 + safe_exp(-(V + 16.0)/5.0))
    m_inf = 1.0 / (1.0 + safe_exp(-(V + 20.0)/12.0))

    ADP_c = max(par.Ac_tot - ATP_c, 0.0)
    MgADP = 0.165 * ADP_c
    ADP3  = 0.135 * ADP_c
    ATP4  = 0.005 * ATP_c
    o_inf = (0.08*(1 + 2*MgADP/17.0) + 0.89*(MgADP/17.0)**2) / (((1 + MgADP/17.0)**2)*(1 + ADP3/26.0 + ATP4/1.0))

    gKCa  = par.gK_Ca * (Ca_c**2)/(par.D**2 + Ca_c**2)
    gKATP = par.gK_ATP * o_inf

    IK    = par.gK  * n      * (V - par.VK)
    ICa   = par.gCa * m_inf  * (V - par.VCa)
    IKCa  = gKCa * (V - par.VK)
    IKATP = gKATP * (V - par.VK)

    dV = -(IK + ICa + IKCa + IKATP) / par.C
    dn = (n_inf - n) / par.tau_n

    Jmem  = -(par.alpha*ICa + par.kPMCA*(Ca_c - par.Ca_bas))
    Jleak = par.pleak * (Ca_er - Ca_c)
    JSERCA = par.kSERCA * Ca_c
    Jer   = Jleak - JSERCA

    dCa_c  = par.fc  * (Jmem + Jer + par.vol_ratio*(JNaCa - Juni))
    dCa_er = -par.fer * par.Vc_over_Ver * Jer

    Jhyd   = (par.khyd * Ca_c + par.khyd_bas) * ATP_c
    dATP_c = Jhyd - par.vol_ratio * JANT

    return np.array([dGi, dG6P, dFBP, dNADHm, dphi_m, dCa_m, dADPm, dV, dn, dCa_c, dCa_er, dATP_c])

# ---------------------------
# Cobelli ODE system
# ---------------------------

def cobelli_rhs(t_min, u, par: CobelliParams, Ix_func: Callable[[float], float], Iu_func: Callable[[float], float]):
    u1, u1p, u2p, u11, u12, u13, u2 = u
    Ge, y2, y3, y4, y5, ex, F1, F2, F3, F4, F5, W, F6, F7 = cobelli_functions(u, par)

    # k02(u1): glucose-dependent transfer from IRP to plasma (smooth, positive)
    k02 = 1e-3 * max(Ge - par.y1_basal, 0.0)

    Ix = Ix_func(t_min)  # mg/min
    Iu = Iu_func(t_min)  # µU/min

    du1  = (F1 - F2) - F3 - F4 - F5 + Ix
    du1p = -par.k21 * u1p + par.k12 * u2p + W
    du2p =  par.k21 * u1p - (par.k12 + k02) * u2p

    du11 = -(par.m01 + par.m21 + par.m31) * u11 + par.m12 * u12 + par.m13 * u13 + Iu
    du12 = -(par.m02 + par.m12) * u12 + par.m21 * u11 + k02 * u2p
    du13 = -par.m13 * u13 + par.m31 * u11

    du2  = -par.h02 * u2 + F7

    return np.array([du1, du1p, du2p, du11, du12, du13, du2])

# ---------------------------
# Coupled system (integrated in ms)
# ---------------------------

def coupled_rhs_ms(t_ms, y, dop: DualOscParams, cop: CobelliParams, Ix_func, Iu_func):
    """
    RHS with time in **milliseconds**.
    - Dual-oscillator is per ms (leave as-is).
    - Cobelli is per minute -> convert to per ms by dividing by 60000.
    """
    t_min = t_ms / 60000.0
    u = y[:7]
    x = y[7:]

    du_min = cobelli_rhs(t_min, u, cop, Ix_func, Iu_func)   # per minute
    du_ms  = du_min / 60000.0                               # per ms

    Ge = u[0] / cop.V1  # mg/100 ml
    dx_ms = dual_oscillator_rhs(t_min, x, Ge, dop)          # per ms

    return np.concatenate([du_ms, dx_ms])

# ---------------------------
# Initialization helpers
# ---------------------------

def initial_cobelli(cop: CobelliParams) -> np.ndarray:
    u1  = cop.y1_basal * cop.V1          # mg
    u11 = cop.y2_basal * cop.V11         # µU
    u2  = cop.y5_basal * cop.V2          # pg

    u13 = cop.m31 * u11 / cop.m13
    u12 = ((cop.m01 + cop.m21 + cop.m31) * u11 - cop.m13 * u13) / cop.m12
    F6_0 = (cop.m02 + cop.m12) * u12 - cop.m21 * u11
    u2p = F6_0 / (0.5 * (1 + tanh(cop.b6 * cop.c6)))
    u1p = (F6_0 + cop.k12 * u2p) / cop.k21

    return np.array([u1, u1p, u2p, u11, u12, u13, u2])

def initial_dual_osc(dop: DualOscParams) -> np.ndarray:
    Gi = 7.0; G6P = 100.0; FBP = 2.0
    NADHm = 0.5; phi_m = 160.0; Ca_m = 0.2
    ADPm = 1.0; V = -60.0; n = 0.05
    Ca_c = 0.1; Ca_er = 200.0; ATP_c = 1400.0
    return np.array([Gi, G6P, FBP, NADHm, phi_m, Ca_m, ADPm, V, n, Ca_c, Ca_er, ATP_c])

# ---------------------------
# Simulation API
# ---------------------------

def simulate(total_minutes: float = 120.0,
             glucose_load_mg_per_kg_min: float = 100.0,
             load_duration_min: float = 3.0,
             BW_kg: float = 70.0):
    dop = DualOscParams()
    cop = CobelliParams(BW_kg=BW_kg)

    load_rate = glucose_load_mg_per_kg_min * BW_kg

    def smooth_step(t_min, t0, t1, rate, tau=0.2):
        s0 = 1.0 / (1.0 + np.exp(-(t_min - t0)/tau))
        s1 = 1.0 / (1.0 + np.exp(-(t_min - t1)/tau))
        return rate * np.clip(s0 - s1, 0.0, 1.0)

    def Ix(t_min):
        return smooth_step(t_min, 0.0, load_duration_min, load_rate, tau=0.2)

    def Iu(t_min):
        return 0.0

    y0 = np.concatenate([initial_cobelli(cop), initial_dual_osc(dop)])

    # Integrate in **milliseconds**
    t_span_ms = (0.0, total_minutes * 60000.0)
    # sample every second for plotting
    t_eval_ms = np.linspace(t_span_ms[0], t_span_ms[1], int(total_minutes * 60))

    sol = solve_ivp(
        fun=lambda tt, yy: coupled_rhs_ms(tt, yy, dop, cop, Ix, Iu),
        t_span=t_span_ms,
        y0=y0,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
        t_eval=t_eval_ms  # <-- Pass t_eval here!
    )

    if not sol.success:
        raise RuntimeError("Integration failed: " + sol.message)

    result = {
        "t_min": sol.t / 60000.0,  # sol.t now contains the evaluation points
        "u": sol.y[:7, :],
        "x": sol.y[7:, :],
        "params": {"dual": dop, "cobelli": cop},
        "description": "Variables: u=[u1,u1p,u2p,u11,u12,u13,u2]; x=[Gi,G6P,FBP,NADHm,phi_m,Ca_m,ADPm,V,n,Ca_c,Ca_er,ATP_c]"
    }
    return result


# convenience metric
def compute_Jo(NADHm, phi_m, par: DualOscParams):
    phi_eff = np.clip(phi_m, 120.0, 200.0)
    return ((par.p4 * NADHm) / (par.p5 + NADHm)) * (1.0 / (1.0 + safe_exp((phi_eff - par.p10)/par.p11)))

def plot_figure6_like(result):
    """Replicate panels (a)-(f): Ca_c, V, FBP, ATP_c, NADHm, Jo."""
    import matplotlib.pyplot as plt
    t = result["t_min"]; x = result["x"]; par = result["params"]["dual"]
    Ca_c = x[9]; V = x[7]; FBP = x[2]; ATP_c = x[11]; NADHm = x[3]; phi_m = x[4]
    Jo = compute_Jo(NADHm, phi_m, par)

    plt.figure(); plt.plot(t, Ca_c); plt.xlabel('Time, t (min)'); plt.ylabel('Free cytosolic Ca$^{2+}$, $C_{a_c}$ (µM)'); plt.title('(a)')
    plt.figure(); plt.plot(t, V);    plt.xlabel('Time, t (min)'); plt.ylabel('Membrane potential, V (mV)'); plt.title('(b)')
    plt.figure(); plt.plot(t, FBP);  plt.xlabel('Time, t (min)'); plt.ylabel('Fructose 1,6-bisphosphate, FBP (µM)'); plt.title('(c)')
    plt.figure(); plt.plot(t, ATP_c);plt.xlabel('Time, t (min)'); plt.ylabel('Cytosolic ATP, ATP$_c$ (µM)'); plt.title('(d)')
    plt.figure(); plt.plot(t, NADHm);plt.xlabel('Time, t (min)'); plt.ylabel('Mitochondrial NADH, NADH$_m$ (mM)'); plt.title('(e)')
    plt.figure(); plt.plot(t, Jo);   plt.xlabel('Time, t (min)'); plt.ylabel('Rate of oxygen consumption, $J_o$ (µM/ms)'); plt.title('(f)')
    plt.show()

# ---------------------------
# Script entry
# ---------------------------

if __name__ == "__main__":
    res = simulate(total_minutes=20, glucose_load_mg_per_kg_min=40.0, load_duration_min=2.0)
    plot_figure6_like(res)
