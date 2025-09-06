# Chew et al. (2009) Dual Oscillator + optional Cobelli coupling
# Paper-faithful defaults (Table A1) + minimal tweaks option to ensure bursting.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Callable, Optional
from scipy.integrate import solve_ivp

# ---------- utilities ----------
def safe_exp(z, clip=50.0):
    return np.exp(np.clip(z, -clip, clip))

FRT_INV_MV = 0.037  # 0.5*F/RT in 1/mV (scaled as needed)


@dataclass
class DualOscParams:
    # Glycolytic (Table A1)
    Vglut: float = 8.0      # mM/s   (paper)
    Kglut: float = 8.0
    Vgk:   float = 0.8      # mM/s   (paper)
    Kgk:   float = 7.0
    ngk:   float = 4.0

    # PFK/GPDH (Table A1)
    kGPDH: float = 0.0003    # µM^(1/2)/ms
    Vmax:  float = 0.005     # µM/ms
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

    # Ca2+ (Table A1)
    C: float = 5300.0
    tau_n: float = 20.0
    gK: float = 2700.0; gCa: float = 1000.0; gK_Ca: float = 300.0; gK_ATP: float = 16000.0
    VK: float = -75.0; VCa: float = 25.0

    D: float = 0.5
    khyd: float = 6e-5
    khyd_bas: float = 1e-5
    alpha: float = 4.5e-6
    kPMCA: float = 0.1

    fc: float = 0.01; fer: float = 0.01; Vc_over_Ver: float = 31.0
    Ca_bas: float = 0.05
    pleak: float = 0.0002; kSERCA: float = 0.4

    # Cytosolic nucleotides
    Ac_tot: float = 2200.0
    AMP_c: float = 400.0

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
    def V1(self):  return self.V1_frac  * self.BW_kg * 10
    @property
    def V11(self): return self.V11_frac * self.BW_kg * 10
    @property
    def V12(self): return self.V12_frac * self.BW_kg * 10
    @property
    def V13(self): return self.V13_frac * self.BW_kg * 10
    @property
    def V2(self):  return self.V2_frac  * self.BW_kg * 10


# ---------------------------
# Helper nonlinear functions (Cobelli)
# ---------------------------

def tanh(x): return np.tanh(x)

def cobelli_functions(u, p: CobelliParams):
    u1, u1p, u2p, u11, u12, u13, u2 = u
    Ge = u1 / p.V1
    y2 = u11 / p.V11; y3 = u12 / p.V12; y4 = u13 / p.V13; y5 = u2 / p.V2
    ex  = Ge - p.y1_basal; e12 = y3 - p.y2_basal; e21 = y5 - p.y5_basal; e13 = y4 - p.y2_basal

    # B: unit processes
    G1 = 0.5 * (1 + tanh(p.b11 * (e21 + p.c11)))
    H1 = 0.5 * (1 - tanh(p.b12 * (e12 + p.c12)) + (1 - tanh(p.b13 * (e12 + p.c13))))
    M1 = 0.5 * (1 - tanh(p.b14 * (ex  + p.c14)))
    F1 = p.a11 * G1 * H1 * M1

    H2 = 0.5 * p.a21 * (1 + tanh(p.b21 * (e12 + p.c21)))
    M2 = 0.5 * p.a22 * (1 + tanh(p.b22 * (ex  + p.c22)))
    F2 = H2 + M2

    # renal excretion term (active only at hyperglycemia)
    if Ge > 252.0:  # mg/dL
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
    # Eqs A.6–A.7
    f13, f23, f41, f42, f43 = par.f13, par.f23, par.f41, par.f42, par.f43
    K1, K2, K3, K4 = par.K1, par.K2, par.K3, par.K4

    def omega(i, j, k, l):
        return (1.0/((f13**i)*(f23**j)*(f41**i)*(f42**j)*(f43**k))) * \
               ((AMP_uM/K1)**i) * ((FBP_uM/K2)**j) * ((F6P_uM/K3)**k) * ((ATP_uM/K4)**l)

    den = sum(omega(i,j,k,l) for i in (0,1) for j in (0,1) for k in (0,1) for l in (0,1))
    num = (1 - par.lam_pfk) * omega(1,1,1,0) + par.lam_pfk * sum(
        omega(i,j,1,l) for i in (0,1) for j in (0,1) for l in (0,1)
    )
    return par.Vmax * (num/den)  # µM/ms


def dual_oscillator_rhs(t_min, x, Ge_mg_per_100ml: float, par: DualOscParams):
    """
    x = [Gi (mM), G6P (µM), FBP (µM), NADHm (mM), phi_m (mV),
         Ca_m (µM), ADPm (mM), V (mV), n, Ca_c (µM), Ca_er (µM), ATP_c (µM)]
    """
    Gi, G6P, FBP, NADHm, phi_m, Ca_m, ADPm, V, n, Ca_c, Ca_er, ATP_c = x

    # A.1–A.3 GLUT2 + GK  (convert paper's mM/s → mM/ms)
    Ge_mM = Ge_mg_per_100ml / 18.0
    Vglut_ms = par.Vglut / 1000.0  # mM/ms
    Vgk_ms   = par.Vgk   / 1000.0  # mM/ms

    Jglut = Vglut_ms * (Ge_mM - Gi) * par.Kglut / ((par.Kglut + Ge_mM) * (par.Kglut + Gi))  # mM/ms
    JGK   = Vgk_ms   * (Gi**par.ngk) / (par.Kgk**par.ngk + Gi**par.ngk)                      # mM/ms
    dGi   = Jglut - JGK

    # A.4–A.7 glycolysis (G6P/FBP in µM; PFK/GPDH in µM/ms)
    F6P   = 0.3 * G6P
    JPFK  = pfk_rate(F6P, FBP, ATP_c, par.AMP_c, par)          # µM/ms
    JGPDH = par.kGPDH * np.sqrt(max(FBP, 0.0))                 # µM/ms
    dG6P  = (JGK * 1000.0) - JPFK                               # mM/ms → µM/ms
    dFBP  = JPFK - 0.5 * JGPDH

    # Mitochondria A.8–A.22 (NADHm, ADPm are mM; use 0.001 to convert µM→mM)
    NADm = max(par.NADm_tot - NADHm, 0.0)
    RATm = max((par.Am_tot - ADPm)/max(ADPm, 1e-12), 0.0)

    JPDH   = ((par.p1*NADm)/(par.p2*NADm + NADHm)) * (Ca_m/(par.p3 + Ca_m)) * (JGPDH + par.JGPDH_bas)
    Jo     = ((par.p4*NADHm)/(par.p5 + NADHm)) * (1.0/(1.0 + safe_exp((phi_m - par.p6)/par.p7)))
    dNADHm = 0.001 * (JPDH - Jo)

    JH_res = ((par.p8*NADHm)/(par.p9 + NADHm)) * (1.0/(1.0 + safe_exp((phi_m - par.p10)/par.p11)))
    JF1F0  = (par.p13/(par.p13 + (par.Am_tot - ADPm))) * (par.p16/(1.0 + safe_exp((par.p14 - phi_m)/par.p15)))
    JH_atp = 3.0 * JF1F0
    JANT   = par.p19 * (RATm/(RATm + par.p20)) * safe_exp(0.5*FRT_INV_MV*phi_m)
    JH_leak = par.p17*phi_m + par.p18

    Juni  = (par.p21*phi_m - par.p22) * Ca_c
    JNaCa = par.p23 * (Ca_m/max(Ca_c, 1e-9)) * safe_exp(par.p24*phi_m)

    dphi_m = (JH_res - JH_atp - JANT - JH_leak - 2.0*Juni - JNaCa) / par.Cm
    dCa_m  = -par.fm * (JNaCa - Juni)
    dADPm  = 0.001 * (JANT - JF1F0)

    # Membrane & Ca2+ (A.23–A.33, A.40–A.45)
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

    dV = -(IK + ICa + IKCa + IKATP) / par.C
    dn = (n_inf - n) / par.tau_n

    Jmem   = -(par.alpha*ICa + par.kPMCA*(Ca_c - par.Ca_bas))
    Jleak  = par.pleak * (Ca_er - Ca_c)
    JSERCA = par.kSERCA * Ca_c
    Jer    = Jleak - JSERCA

    dCa_c  = par.fc  * (Jmem + Jer + par.vol_ratio*(JNaCa - Juni))
    dCa_er = -par.fer * par.Vc_over_Ver * Jer

    # Cytosolic ATP
    Jhyd   = (par.khyd*Ca_c + par.khyd_bas) * ATP_c
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
    Gi = 7.0; G6P = 50.0; FBP = 10.0
    NADHm = 0.5; phi_m = 160.0; Ca_m = 0.2
    ADPm = 1.0; V = -60.0; n = 0.05
    Ca_c = 0.1; Ca_er = 200.0; ATP_c = 1200.0
    return np.array([Gi, G6P, FBP, NADHm, phi_m, Ca_m, ADPm, V, n, Ca_c, Ca_er, ATP_c])


# ---------------------------
# Burn-in helper
# ---------------------------

def burn_in_clamped(x0, dop, Ge_mM: float, minutes: float, rtol=2e-5, atol=5e-8, max_step=200.0):
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
    # GK in mM/ms from paper's mM/s
    JGK_mMms = (dop.Vgk/1000.0) * (Gi**dop.ngk) / (dop.Kgk**dop.ngk + Gi**dop.ngk)
    JGK = 1000.0 * JGK_mMms  # µM/ms
    JPFK = pfk_rate(F6P, FBP, _rest[-1], dop.AMP_c, dop)  # µM/ms
    JGPDH = dop.kGPDH*np.sqrt(max(FBP, 0.0))              # µM/ms
    return JGK, JPFK, JGPDH


# ---------------------------
# Calibration (kept tiny & local)
# ---------------------------

def calibrate_fluxes_at_baseline(dop: DualOscParams, Ge_mM: float, minutes: float = 8.0):
    # 1) Burn in once to get a reasonable cellular state at baseline Ge
    x0 = burn_in_clamped(initial_dual_osc(dop), dop, Ge_mM, minutes)

    # 2) Evaluate JGK at that state (in µM/ms), but evaluate JPFK with FBP clamped to ~5 µM
    Gi, G6P, *_ = x0
    ATP_c = x0[11]
    F6P = 0.3*G6P

    JGK_uMms = compute_JGK(Gi, dop) * 1000.0
    FBP_ref  = 5.0  # µM target for calibration (low-µM regime)
    JPFK_uMms = pfk_rate(F6P, FBP_ref, ATP_c, dop.AMP_c, dop)

    # 3) Rescale only PFK Vmax so that JPFK ≈ JGK at that reference point
    scale = JGK_uMms / max(JPFK_uMms, 1e-12)
    dop = replace(dop, Vmax=dop.Vmax * scale)

    # Optional: quick report
    print(f"[Calib @ {Ge_mM:.1f} mM] target FBP={FBP_ref:.1f} µM, "
          f"JGK={JGK_uMms:.3e}, JPFK(clamped)→{(JPFK_uMms*scale):.3e} (µM/ms)")

    return dop



# ---------------------------
# Simulation API
# ---------------------------

def simulate(total_minutes: float = 120.0,
             burn_in_minutes: float = 30.0,
             mode: str = "clamped",
             # clamped glucose options
             profile: str = "sweep",   # "step" | "sweep" | "ramp"
             ge_baseline_mM: float = 11.0,
             ge_step_mM: float = 15.0,
             ge_step_time_min: float = 10.0,
             ge_ramp_lo_mM: float = 8.0,
             ge_ramp_hi_mM: float = 17.0,
             ge_sweep_period_min: float = 80.0,
             # paper-faithful vs exploratory
             strict: bool = True,
             burst_helper_tweaks: bool = True,
             # numerical
             rtol: float = 2e-5, atol: float = 5e-8, max_step: float = 200.0,
             # kick
             kick: bool = True, kick_FBP_uM: float = 2.0, kick_V_mV: float = -2.0,
             # Cobelli extras
             glucose_load_mg_per_kg_min: float = 100.0,
             load_duration_min: float = 3.0,
             BW_kg: float = 70.0,
             dual_overrides: Optional[dict] = None):

    dop = DualOscParams()

    if burst_helper_tweaks and strict:
        dop = replace(dop, gK_ATP=dop.gK_ATP*0.9, kPMCA=dop.kPMCA*0.85, Ac_tot=dop.Ac_tot*0.9)

    if dual_overrides:
        dop = replace(dop, **dual_overrides)

    # Calibrate glycolytic balance at baseline Ge
    dop = calibrate_fluxes_at_baseline(dop, ge_baseline_mM, minutes=8.0)
    # --- Calibrate kGPDH to pin FBP near a physiological target ---
    # Brief burn-in at baseline just to *measure* baseline JPFK with the new Vmax
    x_probe = burn_in_clamped(initial_dual_osc(dop), dop, ge_baseline_mM, minutes=2.0,
                            rtol=rtol, atol=atol, max_step=max_step)
    JGK0_uMms, JPFK0_uMms, _ = glycolysis_diagnostics(x_probe, dop, ge_baseline_mM)

    # Choose an FBP target in the tens of µM (typical modeling range)
    FBP_target = 20.0   # µM (you can try 10–50 µM)
    JPFK_base  = JPFK0_uMms

    # From dFBP = JPFK - 0.5*kGPDH*sqrt(FBP) ≈ 0 at baseline  ⇒  kGPDH = 2*JPFK_base/sqrt(FBP_target)
    k_needed = 2.0 * JPFK_base / np.sqrt(FBP_target)
    dop = replace(dop, kGPDH=k_needed)
    print(f"[kGPDH calibration] kGPDH = {k_needed:.3e} µM^(1/2)/ms  (FBP_target={FBP_target:g} µM)")


    # burn-in (clamped)
    x0 = initial_dual_osc(dop)
    x0 = burn_in_clamped(x0, dop, ge_baseline_mM, burn_in_minutes, rtol, atol, max_step)

    # diagnostics at end of burn-in
    JGK0, JPFK0, JGPDH0 = glycolysis_diagnostics(x0, dop, ge_baseline_mM)
    print(f"[Diagnostics @ burn-in end] JGK={JGK0:.4e}  JPFK={JPFK0:.4e}  JGPDH={JGPDH0:.4e}  (µM/ms)")

    if kick:
        x0 = x0.copy()
        x0[2] += kick_FBP_uM  # FBP
        x0[7] += kick_V_mV    # V

    # clamped plasma glucose profile
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
            xphase = (t_min % period)/period
            tri = 2.0*abs(xphase - 0.5)  # 1→0→1
            Ge_mM = ge_ramp_lo_mM + (1.0 - tri)*(ge_ramp_hi_mM - ge_ramp_lo_mM)
        else:
            Ge_mM = ge_baseline_mM
        return Ge_mM*18.0  # mg/dL

    if mode.lower() == "clamped":
        def rhs(tt, xx): return dual_oscillator_rhs(tt/60000.0, xx, Ge_profile(tt/60000.0), dop)
        t_span_ms = (0.0, total_minutes*60000.0)
        t_eval_ms = np.linspace(*t_span_ms, int(total_minutes*60*10))  # 10 Hz in minutes
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

def compute_JGK(Gi_mM, par):
    return (par.Vgk/1000.0) * (Gi_mM**par.ngk) / (par.Kgk**par.ngk + Gi_mM**par.ngk)  # mM/ms


def compute_o_inf(ATP_c_uM, ADP_c_uM):
    MgADP = 0.165*ADP_c_uM; ADP3 = 0.135*ADP_c_uM; ATP4 = 0.005*ATP_c_uM
    return (0.08*(1 + 2*MgADP/17.0) + 0.89*(MgADP/17.0)**2) / (((1 + MgADP/17.0)**2)*(1 + ADP3/26.0 + ATP4/1.0))

def debug_plots_clamped(result):
    import matplotlib.pyplot as plt
    t = result["t_min"]; x = result["x"]; par = result["params"]["dual"]
    Ge_profile = result["params"].get("Ge_profile", lambda tt: 11.0*18.0)
    Ge = np.array([Ge_profile(tt) for tt in t]); Ge_mM = Ge/18.0
    Gi, FBP, ATP_c = x[0], x[2], x[11]
    ADP_c = np.maximum(par.Ac_tot - ATP_c, 0.0)
    JGK = compute_JGK(Gi, par)*1000.0  # µM/ms
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

def _extract_mode_params(result):
    mode = result["params"].get("mode", "clamped")
    dual = result["params"]["dual"]
    cob = result["params"].get("cobelli", None)
    return mode, dual, cob

def _external_glucose_series(result):
    t = result["t_min"]
    mode, _dual, cob = _extract_mode_params(result)
    if mode == "clamped":
        Ge_profile = result["params"]["Ge_profile"]
        Ge_mM = np.array([Ge_profile(tt) for tt in t]) / 18.0
        return Ge_mM
    else:
        u = result["u"]
        Ge_mgdL = u[0] / cob.V1
        return Ge_mgdL / 18.0

def _compute_cell_timeseries(result):
    t = result["t_min"]; x = result["x"]; par = result["params"]["dual"]
    Gi = x[0]; G6P = x[1]; FBP = x[2]; NADHm = x[3]; phi_m = x[4]
    V = x[7]; Ca_c = x[9]; ATP_c = x[11]
    Jo = compute_Jo(NADHm, phi_m, par)
    JGK_uMms = compute_JGK(Gi, par) * 1000.0
    JPFK = pfk_rate(0.3*G6P, FBP, ATP_c, par.AMP_c, par)
    return {"Gi_mM": Gi, "JGK_uMms": JGK_uMms, "JPFK_uMms": JPFK,
            "FBP_uM": FBP, "Ca_c_uM": Ca_c, "V_mV": V, "ATP_c_uM": ATP_c,
            "NADHm_mM": NADHm, "Jo_uMms": Jo}

def plot_body_to_cellular_linked(result, save_path=None, figsize=(10, 12)):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    t = result["t_min"]
    mode, dual, cob = _extract_mode_params(result)
    Ge_mM = _external_glucose_series(result)
    series = _compute_cell_timeseries(result)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(7, 1, height_ratios=[1.2, 1.1, 1, 1, 1, 1, 1], hspace=0.25)

    ax_ge = fig.add_subplot(gs[0, 0])
    ax_ge.plot(t, Ge_mM, lw=1.8)
    ax_ge.set_ylabel("Plasma glucose (mM)")
    ax_ge.set_title("Whole-body → Cellular linkage")

    ax_j = fig.add_subplot(gs[1, 0], sharex=ax_ge)
    ax_j.plot(t, series["JGK_uMms"], lw=1.5, label=r"$J_{GK}$")
    ax_j.plot(t, series["JPFK_uMms"], lw=1.2, alpha=0.8, label=r"$J_{PFK}$")
    ax_j.set_ylabel("Flux (µM/ms)")
    ax_j.legend(loc="upper right", ncol=2, fontsize=9)

    ax_f = fig.add_subplot(gs[2, 0], sharex=ax_ge)
    ax_f.plot(t, series["FBP_uM"], lw=1.5)
    ax_f.set_ylabel("FBP (µM)")

    ax_c = fig.add_subplot(gs[3, 0], sharex=ax_ge)
    ax_c.plot(t, series["Ca_c_uM"], lw=1.5)
    ax_c.set_ylabel(r"$Ca_c$ (µM)")

    ax_v = fig.add_subplot(gs[4, 0], sharex=ax_ge)
    ax_v.plot(t, series["V_mV"], lw=1.2)
    ax_v.set_ylabel("V (mV)")

    ax_atp = fig.add_subplot(gs[5, 0], sharex=ax_ge)
    ax_atp.plot(t, series["ATP_c_uM"], lw=1.5)
    ax_atp.set_ylabel(r"$ATP_c$ (µM)")

    ax_n = fig.add_subplot(gs[6, 0], sharex=ax_ge)
    ax_n.plot(t, series["NADHm_mM"], lw=1.4, label=r"$NADH_m$ (mM)")
    ax_n2 = ax_n.twinx()
    ax_n2.plot(t, series["Jo_uMms"], lw=1.0, linestyle="--", label=r"$J_o$ (µM/ms)")
    ax_n.set_ylabel(r"$NADH_m$ (mM)")
    ax_n2.set_ylabel(r"$J_o$ (µM/ms)")
    ax_n.set_xlabel("Time (min)")

    for ax in [ax_ge, ax_j, ax_f, ax_c, ax_v, ax_atp, ax_n]:
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_chew2009_style(result, save_path=None, skip_fast=20, skip_slow=2):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t = result["t_min"]
    Ge_mM = _external_glucose_series(result)
    series = _compute_cell_timeseries(result)

    def ds(arr, skip):  # downsample helper
        return arr[::skip] if skip > 1 else arr

    # downsample
    t_fast = ds(t, skip_fast)
    t_slow = ds(t, skip_slow)

    Ge_ds   = ds(Ge_mM, skip_slow)
    JGK_ds  = ds(series["JGK_uMms"], skip_slow)
    JPFK_ds = ds(series["JPFK_uMms"], skip_slow)
    FBP_ds  = ds(series["FBP_uM"], skip_slow)
    ATP_ds  = ds(series["ATP_c_uM"], skip_slow)

    Ca_ds   = ds(series["Ca_c_uM"], skip_fast)
    V_ds    = ds(series["V_mV"], skip_fast)
    NADH_ds = ds(series["NADHm_mM"], skip_fast)
    Jo_ds   = ds(series["Jo_uMms"], skip_fast)

    fig = plt.figure(figsize=(10.0, 12.0), constrained_layout=True)
    gs = gridspec.GridSpec(7, 1, figure=fig,
                           height_ratios=[1.2, 1.1, 1, 1, 1, 1, 1])

    ax_ge = fig.add_subplot(gs[0, 0])
    ax_ge.plot(t_slow, Ge_ds, lw=1.6)
    ax_ge.set_ylabel("Glucose (mM)")
    ax_ge.set_title("Whole-body → Cellular linkage")
    ax_ge.grid(True, alpha=0.2)

    ax_flux = fig.add_subplot(gs[1, 0], sharex=ax_ge)
    ax_flux.plot(t_slow, JGK_ds,  lw=1.5, label=r"$J_{GK}$")
    ax_flux.plot(t_slow, JPFK_ds, lw=1.2, alpha=0.9, label=r"$J_{PFK}$")
    ax_flux.set_ylabel("Flux (µM/ms)")
    ax_flux.grid(True, alpha=0.2)
    ax_flux.legend(loc="upper right", fontsize=9, ncol=2)

    ax_fbp = fig.add_subplot(gs[2, 0], sharex=ax_ge)
    ax_fbp.plot(t_slow, FBP_ds, lw=1.4)
    ax_fbp.set_ylabel("FBP (µM)")
    ax_fbp.grid(True, alpha=0.2)

    ax_ca = fig.add_subplot(gs[3, 0], sharex=ax_ge)
    ax_ca.plot(t_fast, Ca_ds, lw=1.4)
    ax_ca.set_ylabel(r"$Ca_c$ (µM)")
    ax_ca.grid(True, alpha=0.2)

    ax_v = fig.add_subplot(gs[4, 0], sharex=ax_ge)
    ax_v.plot(t_fast, V_ds, lw=1.2)
    ax_v.set_ylabel("V (mV)")
    ax_v.grid(True, alpha=0.2)

    ax_atp = fig.add_subplot(gs[5, 0], sharex=ax_ge)
    ax_atp.plot(t_slow, ATP_ds, lw=1.4)
    ax_atp.set_ylabel(r"$ATP_c$ (µM)")
    ax_atp.grid(True, alpha=0.2)

    ax_n = fig.add_subplot(gs[6, 0], sharex=ax_ge)
    ax_n.plot(t_fast, NADH_ds, lw=1.3, label=r"$NADH_m$ (mM)")
    ax_n.set_ylabel(r"$NADH_m$ (mM)")
    ax_n.grid(True, alpha=0.2)
    ax_n.set_xlabel("Time (min)")

    ax_n2 = ax_n.twinx()
    ax_n2.plot(t_fast, Jo_ds, lw=1.0, linestyle="--", label=r"$J_o$ (µM/ms)")
    ax_n2.set_ylabel(r"$J_o$ (µM/ms)")

    lines_left,  labels_left  = ax_n.get_legend_handles_labels()
    lines_right, labels_right = ax_n2.get_legend_handles_labels()
    if lines_left or lines_right:
        ax_n.legend(lines_left + lines_right, labels_left + labels_right,
                    loc="upper right", fontsize=9)

    if save_path:
        fig.savefig(save_path.replace(".png", "_linked.png"), dpi=180)
    plt.show()


def oscillation_score(result):
    import numpy as np
    t = result["t_min"]; x = result["x"]
    V = x[7, :]; Ca = x[9, :]
    mask = t > (0.5 * t[-1])
    V = V[mask]; Ca = Ca[mask]
    stdV = float(np.std(V)); stdCa = float(np.std(Ca))
    score = stdV + 50.0 * stdCa
    print(f"[OscScore] std(V)={stdV:.3f}, std(Ca)={stdCa:.4f}, total={score:.3f}")
    return score


# ---------------------------
# Script entry (example)
# ---------------------------

if __name__ == "__main__":
    dual_overrides = {
        "kGPDH": 6e-3,
        "gK_ATP": 14000,
        "kPMCA": 0.07,
        "Ac_tot": 1400,    # ↓ cytosolic adenine pool
        "vol_ratio": 0.1,  # ↑ coupling mito→cyto ATP
    }

    """res = simulate(
        total_minutes=120,
        burn_in_minutes=30,
        mode="clamped",
        profile="sweep",
        ge_ramp_lo_mM=8.0,
        ge_ramp_hi_mM=17.0,
        dual_overrides=dual_overrides,
        strict=True,
        burst_helper_tweaks=True,
        kick=True,
    )"""

    res2 = simulate(
        total_minutes=120,
        burn_in_minutes=30,
        mode="cobelli",
        glucose_load_mg_per_kg_min=100.0,
        load_duration_min=3.0,
        BW_kg=70.0,
        dual_overrides={
            "kGPDH": 8e-3,
            "JGPDH_bas": 5e-4,
            "gK_ATP": 16000,
            "kPMCA": 0.12,
            "kSERCA": 0.45,
            "Ac_tot": 2200,
        },
        strict=True,
        kick=True,
        kick_FBP_uM=1.0,
        kick_V_mV=-1.0,
    )

    plot_chew2009_style(res2, skip_fast=20, skip_slow=2)
    #plot_chew2009_style(res,  skip_fast=20, skip_slow=2)
    oscillation_score(res2)

    import matplotlib.pyplot as plt
    plt.show()
