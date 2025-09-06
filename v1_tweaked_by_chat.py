# Chew et al. (2009) Dual Oscillator + optional Cobelli coupling
# Fast & stable: parameters balanced for oscillations, per-ms units in the cell model.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Callable

try:
    from scipy.integrate import solve_ivp
except Exception as e:
    raise RuntimeError("This script requires SciPy. Please `pip install scipy`.\n" + str(e))

# ---------- utilities ----------
def safe_exp(z, clip=50.0):
    return np.exp(np.clip(z, -clip, clip))

FRT_INV_MV = 0.037  # 1/mV at 37°C

# ---------------------------
# Parameters (Appendix A & B, tuned for bursting)
# ---------------------------

@dataclass
class DualOscParams:
    # Glycolytic feed (per ms, *small* so JGK matches PFK/GPDH in µM/ms)
    # These values give JGK ≈ few×10^-3 µM/ms at ~11 mM glucose after burn-in.
    Vglut: float = 7.5e-05      # mM/ms  (≈ 0.3 × 2.5e-4)
    Kglut: float = 7.0
    Vgk:   float = 7.5e-06      # mM/ms  (≈ 0.3 × 2.5e-5)
    Kgk:   float = 7.0
    ngk:   float = 4.0

    # PFK/GPDH (A.4–A.7)
    # Bring PFK into the same decade as JGK; add a bit more sink through GPDH.
    Vmax:  float = 12.0         # µM/ms
    kGPDH: float = 0.001        # µM/ms^0.5
    lam_pfk: float = 0.06
    K1: float = 30.0
    K2: float = 1.0
    K3: float = 50000.0
    K4: float = 220.0
    f13: float = 0.02
    f23: float = 0.2
    f41: float = 20.0
    f42: float = 20.0
    f43: float = 20.0

    # Mitochondria (Table A1)
    p1: float = 400.0;  p2: float = 1.0;   p3: float = 0.01
    p4: float = 0.6;    p5: float = 0.1
    p6: float = 177.0;  p7: float = 5.0
    p8: float = 7.0;    p9: float = 0.1
    p10: float = 177.0; p11: float = 5.0
    p13: float = 10.0;  p14: float = 190.0; p15: float = 8.5
    p16: float = 35.0
    p17: float = 0.002
    p18: float = -0.03
    p19: float = 0.35
    p20: float = 2.0
    p21: float = 0.04
    p22: float = 1.1
    p23: float = 0.01
    p24: float = 0.016
    JGPDH_bas: float = 0.0005
    fm: float = 0.01
    NADm_tot: float = 10.0
    Am_tot: float = 15.0
    Cm: float = 1.8

    # Electrical / Ca2+
    C: float = 5300.0
    tau_n: float = 20.0
    gK: float = 2700.0
    gCa: float = 1000.0
    gK_Ca: float = 300.0         # back to paper value
    gK_ATP: float = 15000.0      # pS: window where KATP swings matter
    VK: float = -75.0
    VCa: float = 25.0
    D: float = 0.5               # µM (sets gKCa sensitivity)

    # Ca handling & ATP hydrolysis (physically correct ATP balance)
    khyd: float = 6e-5
    khyd_bas: float = 1e-4
    alpha: float = 4.5e-6
    kPMCA: float = 0.1

    fc: float = 0.01
    fer: float = 0.01
    Vc_over_Ver: float = 31.0
    Ca_bas: float = 0.05
    pleak: float = 0.0002
    kSERCA: float = 0.4

    # Cytosolic nucleotides
    Ac_tot: float = 2300.0       # µM
    AMP_c: float = 450.0         # µM (helps PFK activation)

    # Geometry
    vol_ratio: float = 0.07


@dataclass
class CobelliParams:
    BW_kg: float = 70.0
    V1_frac: float = 0.20; V11_frac: float = 0.045; V12_frac: float = 0.031; V13_frac: float = 0.106; V2_frac: float = 0.20
    y1_basal: float = 91.5; y2_basal: float = 11.0; y5_basal: float = 75.0
    a11: float = 6.727; b11: float = 2.1451; c11: float = -0.854
    b12: float = 1.18e-2; c12: float = -100.0; b13: float = 0.15; c13: float = 9.8; b14: float = 5.9e-2; c14: float = -20.0
    a21: float = 0.56; b21: float = 1.54e-2; c21: float = -172.0
    a22: float = 0.105; b22: float = 0.1; c22: float = -25.0
    a31: float = 7.14e-6; a32: float = -9.15e-2; b31: float = 0.143; c31: float = -2.52e4
    a41: float = 3.45; b41: float = 1.4e-2; c41: float = -146.0; b42: float = 1.4e-2; c42: float = 0.0
    a51: float = 8.4e-2; a52: float = 3.22e-4; a53: float = 3.29e-2; b51: float = 2.78e-2; c51: float = 91.5
    k12: float = 3.128e-2; k21: float = 4.34e-3
    aw: float = 0.74676; a6: float = 1.0; bw: float = 1.09e-2; cw: float = -175.0; b6: float = 5e-2; c6: float = -44.0
    m01: float = 0.125; m02: float = 0.185; m12: float = 0.209; m13: float = 0.02; m21: float = 0.268; m31: float = 0.042
    a71: float = 2.3479; b71: float = 6.86e-4; c71: float = 99.246; b72: float = 3e-2; c72: float = 40.0; h02: float = 0.086643
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
    Ge = u1 / p.V1
    y2 = u11 / p.V11; y3 = u12 / p.V12; y4 = u13 / p.V13; y5 = u2 / p.V2
    ex  = Ge - p.y1_basal; e12 = y3 - p.y2_basal; e21 = y5 - p.y5_basal; e13 = y4 - p.y2_basal

    G1 = 0.5 * (1 + tanh(p.b11 * (e21 + p.c11)))
    H1 = 0.5 * (1 - tanh(p.b12 * (e12 + p.c12)) + (1 - tanh(p.b13 * (e12 + p.c13))))
    M1 = 0.5 * (1 - tanh(p.b14 * (ex  + p.c14)))
    F1 = p.a11 * G1 * H1 * M1

    H2 = 0.5 * p.a21 * (1 + tanh(p.b21 * (e12 + p.c21)))
    M2 = 0.5 * p.a22 * (1 + tanh(p.b22 * (ex  + p.c22)))
    F2 = H2 + M2

    if u1 > 2.52e4:
        M31 = 0.5 * (1 + tanh(p.b31 * (u1 + p.c31)))
        M32 = p.a31 * u1 + p.a32
        F3 = M31 * M32
    else:
        F3 = 0.0

    H4 = 0.5 * (1 + tanh(p.b41 * (e13 + p.c41)))
    M4 = 0.5 * (1 + tanh(p.b42 * (ex  + p.c42)))
    F4 = p.a41 * H4 * M4

    M51 = p.a51 * tanh(p.b51 * (ex + p.c51))
    M52 = p.a52 * ex + p.a53
    F5 = M51 + M52

    W  = 0.5 * p.aw * (1 + tanh(p.bw * (ex + p.cw)))
    F6 = 0.5 * p.a6 * (1 + tanh(p.b6 * (ex + p.c6))) * u2p

    H7 = 0.5 * (1 - tanh(p.b71 * (e13 + p.c71)))
    M7 = 0.5 * (1 - tanh(p.b72 * (ex  + p.c72)))
    F7 = p.a71 * H7 * M7

    return Ge, y2, y3, y4, y5, ex, F1, F2, F3, F4, F5, W, F6, F7

# ---------------------------
# Dual Oscillator pieces (Appendix A)
# ---------------------------

def pfk_rate(F6P_uM, FBP_uM, ATP_uM, AMP_uM, par: DualOscParams):
    f13, f23, f41, f42, f43 = par.f13, par.f23, par.f41, par.f42, par.f43
    K1, K2, K3, K4 = par.K1, par.K2, par.K3, par.K4
    def omega(i, j, k, l):
        return (1.0/((f13**i)*(f23**j)*(f41**i)*(f42**j)*(f43**k))) * \
               ((AMP_uM/K1)**i) * ((FBP_uM/K2)**j) * ((F6P_uM/K3)**k) * ((ATP_uM/K4)**l)
    den = sum(omega(i,j,k,l) for i in (0,1) for j in (0,1) for k in (0,1) for l in (0,1))
    num = (1 - par.lam_pfk) * omega(1,1,1,0) + par.lam_pfk * sum(omega(i,j,1,l) for i in (0,1) for j in (0,1) for l in (0,1))
    return par.Vmax * (num/den)  # µM/ms

def dual_oscillator_rhs(t_min, x, Ge_mg_per_100ml: float, par: DualOscParams):
    Gi, G6P, FBP, NADHm, phi_m, Ca_m, ADPm, V, n, Ca_c, Ca_er, ATP_c = x

    # GLUT2 + GK (Gi in mM, fluxes in mM/ms)
    Ge_mM = Ge_mg_per_100ml/18.0
    Jglut = par.Vglut * (Ge_mM - Gi) * par.Kglut / ((par.Kglut + Ge_mM)*(par.Kglut + Gi))
    JGK   = par.Vgk   * (Gi**par.ngk) / (par.Kgk**par.ngk + Gi**par.ngk)
    dGi   = Jglut - JGK

    # Glycolysis (µM/ms)
    F6P   = 0.3 * G6P
    JPFK  = pfk_rate(F6P, FBP, ATP_c, par.AMP_c, par)
    JGPDH = par.kGPDH * np.sqrt(max(FBP, 0.0))
    dG6P  = (JGK*1000.0) - JPFK          # mM→µM
    dFBP  = JPFK - 0.5*JGPDH

    # Mitochondria
    NADm = max(par.NADm_tot - NADHm, 0.0)
    RATm = max((par.Am_tot - ADPm)/max(ADPm, 1e-12), 0.0)
    JPDH = ((par.p1*NADm)/(par.p2*NADm + NADHm)) * (Ca_m/(par.p3 + Ca_m)) * (JGPDH + par.JGPDH_bas)
    Jo   = ((par.p4*NADHm)/(par.p5 + NADHm)) * (1.0/(1.0 + safe_exp((phi_m - par.p6)/par.p7)))
    dNADHm = 0.001*(JPDH - Jo)

    JH_res = ((par.p8*NADHm)/(par.p9 + NADHm)) * (1.0/(1.0 + safe_exp((phi_m - par.p10)/par.p11)))
    JF1F0  = (par.p13/(par.p13 + (par.Am_tot - ADPm))) * (par.p16/(1.0 + safe_exp((par.p14 - phi_m)/par.p15)))
    JH_atp = 3.0*JF1F0
    JANT   = par.p19 * (RATm/(RATm + par.p20)) * safe_exp(0.5*FRT_INV_MV*phi_m)
    JH_leak = par.p17*phi_m + par.p18
    Juni   = (par.p21*phi_m - par.p22) * Ca_c
    JNaCa  = par.p23 * (Ca_m/max(Ca_c, 1e-9)) * safe_exp(par.p24*phi_m)
    dphi_m = (JH_res - JH_atp - JANT - JH_leak - 2.0*Juni - JNaCa)/par.Cm
    dCa_m  = -par.fm*(JNaCa - Juni)
    dADPm  = 0.001*(JANT - JF1F0)

    # Membrane & Ca2+
    n_inf = 1.0/(1.0 + safe_exp(-(V + 16.0)/5.0))
    m_inf = 1.0/(1.0 + safe_exp(-(V + 20.0)/12.0))
    ADP_c = max(par.Ac_tot - ATP_c, 0.0)
    MgADP = 0.165*ADP_c; ADP3 = 0.135*ADP_c; ATP4 = 0.005*ATP_c
    o_inf = (0.08*(1 + 2*MgADP/17.0) + 0.89*(MgADP/17.0)**2) / (((1 + MgADP/17.0)**2)*(1 + ADP3/26.0 + ATP4/1.0))
    gKCa  = par.gK_Ca * (Ca_c**2)/(par.D**2 + Ca_c**2)
    gKATP = par.gK_ATP * o_inf
    IK    = par.gK  * n     * (V - par.VK)
    ICa   = par.gCa * m_inf * (V - par.VCa)
    IKCa  = gKCa * (V - par.VK)
    IKATP = gKATP * (V - par.VK)
    dV = -(IK + ICa + IKCa + IKATP)/par.C
    dn = (n_inf - n)/par.tau_n

    Jmem  = -(par.alpha*ICa + par.kPMCA*(Ca_c - par.Ca_bas))
    Jleak = par.pleak * (Ca_er - Ca_c)
    JSERCA = par.kSERCA*Ca_c
    Jer   = Jleak - JSERCA
    dCa_c  = par.fc  * (Jmem + Jer + par.vol_ratio*(JNaCa - Juni))
    dCa_er = -par.fer * par.Vc_over_Ver * Jer

    # Cytosolic ATP (correct sign)
    Jhyd   = (par.khyd*Ca_c + par.khyd_bas)*ATP_c
    dATP_c = par.vol_ratio*JANT - Jhyd

    return np.array([dGi, dG6P, dFBP, dNADHm, dphi_m, dCa_m, dADPm, dV, dn, dCa_c, dCa_er, dATP_c])

# ---------------------------
# Cobelli ODE system (Appendix B)
# ---------------------------

def cobelli_rhs(t_min, u, par: CobelliParams, Ix_func: Callable[[float], float], Iu_func: Callable[[float], float]):
    u1, u1p, u2p, u11, u12, u13, u2 = u
    Ge, y2, y3, y4, y5, ex, F1, F2, F3, F4, F5, W, F6, F7 = cobelli_functions(u, par)
    k02 = 1e-3*max(Ge - par.y1_basal, 0.0)
    Ix = Ix_func(t_min); Iu = Iu_func(t_min)
    du1  = (F1 - F2) - F3 - F4 - F5 + Ix
    du1p = -par.k21*u1p + par.k12*u2p + W
    du2p =  par.k21*u1p - (par.k12 + k02)*u2p
    du11 = -(par.m01 + par.m21 + par.m31)*u11 + par.m12*u12 + par.m13*u13 + Iu
    du12 = -(par.m02 + par.m12)*u12 + par.m21*u11 + k02*u2p
    du13 = -par.m13*u13 + par.m31*u11
    du2  = -par.h02*u2 + F7
    return np.array([du1, du1p, du2p, du11, du12, du13, du2])

# ---------------------------
# Coupled system (integrated in ms)
# ---------------------------

def coupled_rhs_ms(t_ms, y, dop: DualOscParams, cop: CobelliParams, Ix_func, Iu_func):
    t_min = t_ms/60000.0
    u = y[:7]; x = y[7:]
    du_min = cobelli_rhs(t_min, u, cop, Ix_func, Iu_func)
    du_ms  = du_min/60000.0
    Ge = u[0]/cop.V1
    dx_ms = dual_oscillator_rhs(t_min, x, Ge, dop)
    return np.concatenate([du_ms, dx_ms])

# ---------------------------
# Initialization helpers
# ---------------------------

def initial_cobelli(cop: CobelliParams) -> np.ndarray:
    u1  = cop.y1_basal*cop.V1
    u11 = cop.y2_basal*cop.V11
    u2  = cop.y5_basal*cop.V2
    u13 = cop.m31*u11/cop.m13
    u12 = ((cop.m01 + cop.m21 + cop.m31)*u11 - cop.m13*u13)/cop.m12
    F6_0 = (cop.m02 + cop.m12)*u12 - cop.m21*u11
    u2p = F6_0/(0.5*(1 + tanh(cop.b6*cop.c6)))
    u1p = (F6_0 + cop.k12*u2p)/cop.k21
    return np.array([u1, u1p, u2p, u11, u12, u13, u2])

def initial_dual_osc(dop: DualOscParams) -> np.ndarray:
    Gi = 7.0; G6P = 100.0; FBP = 25.0
    NADHm = 0.5; phi_m = 160.0; Ca_m = 0.2
    ADPm = 1.0; V = -60.0; n = 0.05
    Ca_c = 0.1; Ca_er = 200.0; ATP_c = 1200.0
    return np.array([Gi, G6P, FBP, NADHm, phi_m, Ca_m, ADPm, V, n, Ca_c, Ca_er, ATP_c])

# ---------------------------
# Burn-in helper
# ---------------------------

def burn_in_clamped(x0, dop, Ge_mM: float, minutes: float, rtol=2e-4, atol=2e-7, max_step=100.0):
    def Ge_const(_): return Ge_mM*18.0  # mM → mg/dL
    t_span = (0.0, minutes*60000.0)
    sol = solve_ivp(lambda tt, xx: dual_oscillator_rhs(tt/60000.0, xx, Ge_const(tt/60000.0), dop),
                    t_span, x0, method="BDF", rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError("Burn-in failed: " + sol.message)
    return sol.y[:, -1]

# ---------------------------
# Diagnostics
# ---------------------------

def glycolysis_diagnostics(x, dop: DualOscParams, Ge_mM: float):
    Gi, G6P, FBP, *_rest = x
    F6P = 0.3*G6P
    JGK_mMms = dop.Vgk*(Gi**dop.ngk)/(dop.Kgk**dop.ngk + Gi**dop.ngk)
    JGK = JGK_mMms*1000.0  # µM/ms
    JPFK = pfk_rate(F6P, FBP, _rest[-1], dop.AMP_c, dop)
    JGPDH = dop.kGPDH*np.sqrt(max(FBP,0.0))
    return JGK, JPFK, JGPDH

# ---------------------------
# Simulation API
# ---------------------------

def simulate(total_minutes: float = 60.0,
             burn_in_minutes: float = 30.0,
             mode: str = "clamped",
             # clamped glucose options
             profile: str = "sweep",   # "step" | "sweep" | "ramp"
             ge_baseline_mM: float = 11.0,
             ge_step_mM: float = 15.0,
             ge_step_time_min: float = 10.0,
             ge_ramp_lo_mM: float = 8.0,
             ge_ramp_hi_mM: float = 17.0,
             ge_sweep_period_min: float = 60.0,
             # numerical
             rtol: float = 2e-4, atol: float = 2e-7, max_step: float = 20.0,
             # kick
             kick: bool = True, kick_FBP_uM: float = 2.0, kick_V_mV: float = -2.0,
             # Cobelli extras
             glucose_load_mg_per_kg_min: float = 100.0,
             load_duration_min: float = 3.0,
             BW_kg: float = 70.0):

    dop = DualOscParams()

    # burn-in (clamped at baseline glucose)
    x0 = initial_dual_osc(dop)
    x0 = burn_in_clamped(x0, dop, ge_baseline_mM, burn_in_minutes)

    # print diagnostics at end of burn-in
    JGK0, JPFK0, JGPDH0 = glycolysis_diagnostics(x0, dop, ge_baseline_mM)
    print(f"[Diagnostics @ burn-in end] JGK={JGK0:.4e}  JPFK={JPFK0:.4e}  JGPDH={JGPDH0:.4e}  (µM/ms)")

    if kick:
        x0 = x0.copy()
        x0[2] += kick_FBP_uM  # FBP
        x0[7] += kick_V_mV    # V

    # glucose-time function for clamped mode
    def Ge_profile(t_min: float) -> float:
        if profile == "step":
            Ge_mM = ge_baseline_mM if t_min < ge_step_time_min else ge_step_mM
        elif profile == "ramp":
            if t_min < ge_step_time_min:
                Ge_mM = ge_baseline_mM
            else:
                frac = min((t_min - ge_step_time_min)/10.0, 1.0)
                Ge_mM = ge_baseline_mM + frac*(ge_step_mM - ge_baseline_mM)
        elif profile == "sweep":
            period = ge_sweep_period_min
            x = (t_min % period)/period
            tri = 2.0*abs(x - 0.5)
            Ge_mM = ge_ramp_lo_mM + (1.0 - tri)*(ge_ramp_hi_mM - ge_ramp_lo_mM)
        else:
            Ge_mM = ge_baseline_mM
        return Ge_mM*18.0  # mg/dL

    if mode.lower() == "clamped":
        def rhs(tt, xx): return dual_oscillator_rhs(tt/60000.0, xx, Ge_profile(tt/60000.0), dop)
        t_span_ms = (0.0, total_minutes*60000.0)
        t_eval_ms = np.linspace(*t_span_ms, int(total_minutes*60*10))  # 10 Hz (per minute)
        sol = solve_ivp(rhs, t_span_ms, x0, method="BDF", t_eval=t_eval_ms,
                        rtol=rtol, atol=atol, max_step=max_step)
        if not sol.success:
            raise RuntimeError("Clamped simulation failed: " + sol.message)
        return {
            "t_min": sol.t/60000.0,
            "u": None,
            "x": sol.y,
            "params": {"dual": dop, "Ge_profile": Ge_profile, "mode": "clamped", "profile": profile},
            "description": "Clamped-Ge run; x=[Gi,G6P,FBP,NADHm,phi_m,Ca_m,ADPm,V,n,Ca_c,Ca_er,ATP_c]"
        }

    # ----- Cobelli-coupled mode -----
    cop = CobelliParams(BW_kg=BW_kg)
    load_rate = glucose_load_mg_per_kg_min*BW_kg  # mg/min

    def smooth_step(t_min, t0, t1, rate, tau=0.2):
        s0 = 1.0/(1.0 + np.exp(-(t_min - t0)/tau))
        s1 = 1.0/(1.0 + np.exp(-(t_min - t1)/tau))
        return rate*np.clip(s0 - s1, 0.0, 1.0)

    def Ix(t_min):  # mg/min infusion
        return smooth_step(t_min, 0.0, load_duration_min, load_rate, tau=0.2)

    def Iu(_): return 0.0

    y0 = np.concatenate([initial_cobelli(cop), x0])
    t_span_ms = (0.0, total_minutes*60000.0)
    t_eval_ms = np.linspace(*t_span_ms, int(total_minutes*60*10))

    def rhs_c(tt, yy): return coupled_rhs_ms(tt, yy, dop, cop, Ix, Iu)
    sol = solve_ivp(rhs_c, t_span_ms, y0, method="BDF", t_eval=t_eval_ms,
                    rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError("Cobelli simulation failed: " + sol.message)

    return {
        "t_min": sol.t/60000.0,
        "u": sol.y[:7, :],
        "x": sol.y[7:, :],
        "params": {"dual": dop, "cobelli": cop, "mode": "cobelli"},
        "description": "Cobelli-coupled run; u=[u1,u1p,u2p,u11,u12,u13,u2]; x=[Gi,G6P,FBP,NADHm,phi_m,Ca_m,ADPm,V,n,Ca_c,Ca_er,ATP_c]"
    }

# ---------------------------
# Convenience metrics & plotting
# ---------------------------

def compute_Jo(NADHm, phi_m, par: DualOscParams):
    return ((par.p4*NADHm)/(par.p5 + NADHm)) * (1.0/(1.0 + safe_exp((phi_m - par.p10)/par.p11)))

def compute_JGK(Gi_mM, par: DualOscParams):
    return par.Vgk*(Gi_mM**par.ngk)/(par.Kgk**par.ngk + Gi_mM**par.ngk)  # mM/ms

def compute_o_inf(ATP_c_uM, ADP_c_uM):
    MgADP = 0.165*ADP_c_uM; ADP3 = 0.135*ADP_c_uM; ATP4 = 0.005*ATP_c_uM
    return (0.08*(1 + 2*MgADP/17.0) + 0.89*(MgADP/17.0)**2) / (((1 + MgADP/17.0)**2)*(1 + ADP3/26.0 + ATP4/1.0))

def plot_figure6_like(result):
    import matplotlib.pyplot as plt
    t = result["t_min"]; x = result["x"]; par = result["params"]["dual"]
    Ca_c = x[9]; V = x[7]; FBP = x[2]; ATP_c = x[11]; NADHm = x[3]; phi_m = x[4]
    Jo = compute_Jo(NADHm, phi_m, par)
    plt.figure(); plt.plot(t, Ca_c); plt.xlabel('Time (min)'); plt.ylabel('Ca$_c$ (µM)'); plt.title('(a)')
    plt.figure(); plt.plot(t, V);    plt.xlabel('Time (min)'); plt.ylabel('V (mV)');        plt.title('(b)')
    plt.figure(); plt.plot(t, FBP);  plt.xlabel('Time (min)'); plt.ylabel('FBP (µM)');      plt.title('(c)')
    plt.figure(); plt.plot(t, ATP_c);plt.xlabel('Time (min)'); plt.ylabel('ATP$_c$ (µM)');  plt.title('(d)')
    plt.figure(); plt.plot(t, NADHm);plt.xlabel('Time (min)'); plt.ylabel('NADH$_m$ (mM)'); plt.title('(e)')
    plt.figure(); plt.plot(t, Jo);   plt.xlabel('Time (min)'); plt.ylabel('$J_o$ (µM/ms)'); plt.title('(f)')
    plt.show()

def debug_plots_clamped(result):
    import matplotlib.pyplot as plt
    t = result["t_min"]; x = result["x"]; par = result["params"]["dual"]
    Ge_profile = result["params"].get("Ge_profile", lambda tt: 11.0*18.0)
    Ge = np.array([Ge_profile(tt) for tt in t]); Ge_mM = Ge/18.0
    Gi, FBP, ATP_c = x[0], x[2], x[11]
    ADP_c = np.maximum(par.Ac_tot - ATP_c, 0.0)
    JGK = compute_JGK(Gi, par)*1000.0
    JPFK = pfk_rate(0.3*x[1], FBP, ATP_c, par.AMP_c, par)
    oinf = compute_o_inf(ATP_c, ADP_c)
    fig, axs = plt.subplots(4,1, figsize=(8,8), sharex=True)
    axs[0].plot(t, Ge_mM); axs[0].set_ylabel("Ge (mM)"); axs[0].set_title("Driver & key fluxes")
    axs[1].plot(t, JGK);   axs[1].set_ylabel("JGK (µM/ms)")
    axs[2].plot(t, JPFK);  axs[2].set_ylabel("JPFK (µM/ms)")
    axs[3].plot(t, oinf);  axs[3].set_ylabel("o_inf (KATP)"); axs[3].set_xlabel("Time (min)")
    plt.tight_layout(); plt.show()

def debug_plots_cobelli(result):
    import matplotlib.pyplot as plt
    t = result["t_min"]; u = result["u"]; cop = result["params"]["cobelli"]
    Ge_dL = u[0]/cop.V1; insulin = u[3]/cop.V11; glucagon = u[6]/cop.V2
    fig, axs = plt.subplots(3,1, figsize=(8,7), sharex=True)
    axs[0].plot(t, Ge_dL);    axs[0].set_ylabel("Plasma glucose (mg/dL)")
    axs[1].plot(t, insulin);  axs[1].set_ylabel("Insulin (µU/mL)")
    axs[2].plot(t, glucagon); axs[2].set_ylabel("Glucagon (pg/mL)"); axs[2].set_xlabel("Time (min)")
    plt.tight_layout(); plt.show()

# ---------------------------
# Script entry
# ---------------------------

if __name__ == "__main__":
    # Clamped + sweep is the most reliable to see bursting quickly
    res = simulate(
        total_minutes=60,
        burn_in_minutes=30,
        mode="clamped",
        profile="sweep",
        ge_ramp_lo_mM=8.0,
        ge_ramp_hi_mM=17.0,
        kick=True,
        kick_FBP_uM=2.0,
        kick_V_mV=-2.0
    )
    plot_figure6_like(res)
    debug_plots_clamped(res)

    # # To try the in-vivo (Cobelli) coupling, uncomment:
    # res2 = simulate(total_minutes=60, burn_in_minutes=60, mode="cobelli")
    # plot_figure6_like(res2)
    # debug_plots_cobelli(res2)
