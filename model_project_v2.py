import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
MGDL_PER_mM = 18.01528
def mgdl_to_mM(x): return x / MGDL_PER_mM

# ----------------------------
# Parameters (Tables A1/B1)
# ----------------------------
params_do = {
    # GLUT2 + GK
    'Vglut':8.0,'Kglut':7.0,'Vgk':0.8,'Kgk':7.0,'ngk':4.0,
    # Glycolysis / PFK
    'KGPDH':5e-4,'Vmax':5e-3,'K1':30.0,'K2':1.0,'K3':5e4,'K4':220.0,
    'f13':0.02,'f23':0.2,'f41':20.0,'f42':20.0,'f43':20.0,'lambda_pfk':0.06,
    'AMP_conc':500.0,
    # Mito
    'p1':400.0,'p2':1.0,'p3':0.01,'p4':0.6,'p5':0.1,'p6':177.0,'p7':5.0,
    'p8':7.0,'p9':0.1,'p10':177.0,'p11':5.0,'p13':10.0,'p14':190.0,'p15':8.5,
    'p16':35.0,'p17':0.002,'p18':-0.03,'p19':0.35,'p20':2.0,'p21':0.04,
    'p22':1.1,'p23':0.01,'p24':0.016,'JGPDHbas':5e-4,'fm':0.01,'Cm':1.8,
    'NADm_tot':10.0,'Am_tot':15.0,
    # Electrical / Ca
    'C_elec':5300.0,'gK':2700.0,'gCa':1000.0,'gK_Ca_param':300.0,'gK_ATP_param':16000.0,
    'VK':-75.0,'Vca':25.0,'KD':0.5,'tau_n':20.0,'alpha_ca':4.5e-6,'KPMCA':0.1,
    'Cabas':0.05,'fc':0.01,'Pleak_er':2e-4,'KSERCA':0.4,'Ac_tot':2500.0,'fer':0.01,
    'Vc_Ver_ratio':31.0,'khyd':5e-5,'khyd_bas':5e-5,'k_mito_cytosol_ratio':0.07,
}

params_cb = {
    # Glucose subsystem
    'a11':6.727,'b11':2.1451,'b12':1.18e-2,'b13':0.15,'b14':5.9e-2,
    'c11':-0.854,'c12':-100,'c13':9.8,'c14':-20,
    'a21':0.56,'a22':0.105,'b21':1.54e-2,'b22':0.1,'c21':-172,'c22':-25,
    'a31':7.14e-6,'a32':-9.15e-2,'b31':0.143,'c31':-2.52e4,
    'a41':3.45,'b41':1.4e-2,'b42':1.4e-2,'c41':-146,'c42':0,
    'a51':8.4e-2,'a52':3.22e-4,'a53':3.29e-2,'b51':2.78e-2,'c51':91.5,
    # Insulin subsystem
    'k12':3.128e-2,'k21':4.34e-3,'aw':0.74676,'alpha6':1.0,
    'bw':1.09e-2,'b6':5e-2,'cw':-175,'c6':-44,
    'm01':0.125,'m02':0.185,'m12':0.209,'m13':0.02,'m21':0.268,'m31':0.042,
    # Glucagon subsystem
    'a71':2.3479,'b71':6.86e-4,'b72':3e-2,'c71':99.246,'c72':40,'h02':0.086643,
    # Volumes (L)
    'V1_vol':0.2*70,'V11_vol':0.045*70,'V12_vol':0.031*70,'V13_vol':0.106*70,'V2_vol':0.2*70,
    # Basal concentrations
    'u1_basal_mg_dL':91.5,'u11_basal_microU_mL':11.0,'u2_basal_pg_mL':75.0,
}
# Derived amount baselines
V1_dL = params_cb['V1_vol']*10.0
params_cb['u1_basal']  = params_cb['u1_basal_mg_dL'] * V1_dL
params_cb['u11_basal'] = params_cb['u11_basal_microU_mL'] * params_cb['V11_vol']*1000.0
params_cb['u2_basal']  = params_cb['u2_basal_pg_mL']    * params_cb['V2_vol'] *1000.0

# ----------------------------
# Dual Oscillator fluxes (Appendix A)
# ----------------------------
# ---------- GLUT2 and GK ----------
def Jglut(Ge, Gi, p):
    # Eq. (1): Jglut = Vglut * (Ge - Gi) * Kglut / ((Kglut+Ge)(Kglut+Gi))   [mM/ms]
    Kg = p['Kglut']; V = p['Vglut']
    return V * (Ge - Gi) * Kg / ((Kg + Ge) * (Kg + Gi))

def JGK(Gi, p):
    # Eq. (A.4) Hill GK  [mM/ms]
    Vgk, Kgk, n = p['Vgk'], p['Kgk'], p['ngk']
    x = (Gi / Kgk)**n
    return Vgk * x / (1.0 + x)


# ---------- GPDH ----------
def JGPDH(FBP_uM, p):
    # Eq. (A.5)   JGPDH = kGPDH * sqrt(FBP / 1 μM)   [μM/ms]
    return p['KGPDH'] * np.sqrt(max(FBP_uM, 0.0))


# ---------- PFK (exact MWC form, Eq. A.6–A.7) ----------
def JPFK(F6P_uM, FBP_uM, ATP_uM, p):
    K1,K2,K3,K4 = p['K1'],p['K2'],p['K3'],p['K4']        # μM
    f13,f23,f41,f42,f43 = p['f13'],p['f23'],p['f41'],p['f42'],p['f43']
    lam = p['lambda_pfk']
    AMP_uM = p['AMP_conc']                               # μM (fixed)

    def omega(i,j,k,l):
        num = ( (AMP_uM/K1)**i * (FBP_uM/K2)**j * (F6P_uM/K3)**k * (ATP_uM/K4)**l )
        den = ( (f13**(i*k)) * (f23**(j*k)) * (f41**(i*l)) * (f42**(j*l)) * (f43**(k*l)) )
        return num/den

    # denominator: sum over all 16 terms
    den_sum = sum(omega(i,j,k,l) for i in (0,1) for j in (0,1) for k in (0,1) for l in (0,1))
    # numerator: (1-λ) ω1110 + λ * sum over i,j,l with k=1
    num = (1.0 - lam) * omega(1,1,1,0) + lam * sum(omega(i,j,1,l) for i in (0,1) for j in (0,1) for l in (0,1))

    return p['Vmax'] * num / max(den_sum, 1e-30)         # μM/ms


def dual_rhs(x, t_ms, Ge_of_t, p):
    eps = 1e-12
    Gi,G6P,FBP,NADHm,ADPm,Cam,phi,V,n,Cac,Caer,ATPc = x

    # clamp positives
    NADHm = min(max(NADHm, eps), p['NADm_tot']-eps)
    ADPm  = min(max(ADPm , eps), p['Am_tot']-eps)
    Cam   = max(Cam, eps); Cac=max(Cac,eps); Caer=max(Caer,eps); ATPc=max(ATPc,eps)

    Ge = Ge_of_t(t_ms)  # mM

    # Glycolysis (unit consistency: mM↔μM)
    j_glut = Jglut(Ge, Gi, p)                 # mM/ms
    j_gk   = JGK(Gi, p)                       # mM/ms
    F6P_mM = 0.3*G6P/1000.0                   # if G6P is μM, convert to mM then back to μM for PFK
    F6P_uM = F6P_mM*1000.0
    j_pfk  = JPFK(F6P_uM, FBP, ATPc, p)       # μM/ms
    j_gpdh = JGPDH(FBP, p)                    # μM/ms

    dGi  = j_glut - j_gk
    dG6P = j_gk*1000.0 - j_pfk
    dFBP = j_pfk - 0.5*j_gpdh

    # Mitochondria (Appendix A.8–A.22)
    FRT=0.037
    RATm = (p['Am_tot']-ADPm)/(ADPm+eps)
    JPDH = (p['p1']*NADHm/(p['p2']*(p['NADm_tot']-NADHm)+NADHm)) * (Cam/(p['p3']+Cam)) * (j_gpdh + p['JGPDHbas'])
    JO   = (p['p4']*NADHm/(p['p5']+NADHm)) * (1.0/(1.0+np.exp((phi-p['p6'])/p['p7'])))

    dNADHm = 0.001*(JPDH - JO)

    JH_res = (p['p8']*NADHm/(p['p9']+NADHm)) * (1.0/(1.0+np.exp((phi-p['p10'])/p['p11'])))
    JF1F0  = (p['p13']/(p['p13'] + (p['Am_tot']-ADPm))) * (p['p16']/(1.0+np.exp((p['p14']-phi)/p['p15'])))
    JH_atp = 3.0*JF1F0
    JANT   = p['p19']*(RATm/(RATm+p['p20']))*np.exp(0.5*FRT*phi)
    Juni   = (p['p21']*phi - p['p22'])*Cac
    JNaCa  = p['p23']*(Cam/(Cac+eps))*np.exp(p['p24']*phi)
    JH_leak= p['p17']*phi + p['p18']
    dphi   = (JH_res - JH_atp - JANT - JH_leak - JNaCa - 2.0*Juni)/p['Cm']
    dCam   = -p['fm']*(JNaCa - Juni)
    dADPm  = 0.001*(JANT - JF1F0)

    # Electrical + Ca handling (Appendix A.23–A.45)
    VK,VCa = p['VK'],p['Vca']
    gK,gCa = p['gK'],p['gCa']
    gKCa   = p['gK_Ca_param']*(Cac**2)/(p['KD']**2 + Cac**2)
    ADPc = max(p['Ac_tot'] - ATPc, eps)
    MgADP = 0.165*ADPc; ADP3m = 0.135*ADPc; ATP4 = 0.005*ATPc
    o_inf = (0.08*(1+2*MgADP/17.0) + 0.89*(MgADP/17.0)**2)/((1+MgADP/17.0)**2*(1+ADP3m/26.0+ATP4))
    gKATP = p['gK_ATP_param']*o_inf
    n_inf = 1.0/(1.0+np.exp(-(V+16.0)/5.0))
    m_inf = 1.0/(1.0+np.exp(-(V+20.0)/12.0))
    IK   = gK*n*(V-VK); ICa=gCa*m_inf*(V-VCa); IKCa=gKCa*(V-VK); IKATP=gKATP*(V-VK)
    dV = -(IK + ICa + IKCa + IKATP)/p['C_elec']
    dn = (n_inf - n)/p['tau_n']

    Jmem = -(p['alpha_ca']*ICa + p['KPMCA']*(Cac - p['Cabas']))
    Jer  = p['Pleak_er']*(Caer - Cac) - p['KSERCA']*Cac
    Jm   = (JNaCa - Juni)
    dCac  = p['fc']*(Jmem + Jer + p['k_mito_cytosol_ratio']*Jm)
    dCaer = -p['fer']*((1.0/p['Vc_Ver_ratio'])*Jer)

    Jhyd = (p['khyd']*Cac + p['khyd_bas'])*ATPc
    dATPc = Jhyd - p['k_mito_cytosol_ratio']*JANT*1000.0

    return np.array([dGi,dG6P,dFBP,dNADHm,dADPm,dCam,dphi,dV,dn,dCac,dCaer,dATPc])




# ----------------------------
# Cobelli model (Appendix B)
# ----------------------------
def k02_of_u1(u1):  # simple monotone dependence (you can replace with the paper's fit if you have it)
    return 1e-4 + 1e-9*u1

def cobelli_rhs(u, t_min, p, Ix_g_min=None, Iu_u_min=None):
    u1,u1p,u2p,u11,u12,u13,u2 = u

    Ge_mgdl = u1/(p['V1_vol']*10.0)
    ex  = Ge_mgdl - p['u1_basal_mg_dL']
    e12 = (u12/(p['V12_vol']*1000.0)) - p['u11_basal_microU_mL']
    e13 = (u13/(p['V13_vol']*1000.0)) - p['u11_basal_microU_mL']
    e21 = (u2 /(p['V2_vol'] *1000.0)) - p['u2_basal_pg_mL']

    G1 = 0.5*(1+np.tanh(p['b11']*(e21+p['c11'])))
    H1 = 0.5*(1-np.tanh(p['b12']*(e12+p['c12'])) + (1-np.tanh(p['b13']*e12+p['c13'])))
    M1 = 0.5*(1-np.tanh(p['b14']*(ex+p['c14'])))
    F1 = p['a11']*G1*H1*M1             # g/min

    H2 = 0.5*p['a21']*(1+np.tanh(p['b21']*(e12+p['c21'])))
    M2 = 0.5*p['a22']*(1+np.tanh(p['b22']*(ex+p['c22'])))
    F2 = H2 + M2                        # g/min

    if u1 > 2.52e4:
        M31 = 0.5*(1+np.tanh(p['b31']*(u1+p['c31'])))
        M32 = p['a31']*u1 + p['a32']
        F3 = M31*M32                    # g/min
    else:
        F3 = 0.0

    H4 = 0.5*(1+np.tanh(p['b41']*(e13+p['c41'])))
    M4 = 0.5*(1+np.tanh(p['b42']*(ex+p['c42'])))
    F4 = p['a41']*H4*M4                 # g/min

    M51 = p['a51']*np.tanh(p['b51']*(ex+p['c51']))
    M52 = p['a52']*ex + p['a53']
    F5 = M51 + M52                      # g/min

    W  = 0.5*p['aw']*(1+np.tanh(p['bw']*(ex+p['cw'])))  # µU/min
    F6 = 0.5*p['alpha6']*(1+np.tanh(p['b6']*(ex+p['c6']))) * u2p  # µU/min

    H7 = 0.5*(1 - np.tanh(p['b71']*(e13+p['c71'])))
    M7 = 0.5*(1 - np.tanh(p['b72']*(ex+p['c72'])))
    F7 = p['a71']*H7*M7                 # µg/min

    Ix_g = 0.0 if Ix_g_min is None else Ix_g_min(t_min)   # g/min
    Iu   = 0.0 if Iu_u_min is None else Iu_u_min(t_min)   # µU/min

    g_to_mg = 1000.0
    du1  = g_to_mg*(F1 - F2) - g_to_mg*F3 - g_to_mg*F4 - g_to_mg*F5 + g_to_mg*Ix_g
    du1p = -p['k21']*u1p + p['k12']*u2p + W
    du2p = p['k21']*u1p - (p['k12'] + k02_of_u1(u1))*u2p
    du11 = -(p['m01']+p['m21']+p['m31'])*u11 + p['m12']*u12 + p['m13']*u13 + Iu
    du12 = -(p['m02']+p['m12'])*u12 + p['m21']*u11 + k02_of_u1(u1)*u2p
    du13 = -p['m13']*u13 + p['m31']*u11
    du2  = -p['h02']*u2 + F7*1000.0     # µg/min -> pg/min

    return np.array([du1,du1p,du2p,du11,du12,du13,du2])

# ----------------------------
# Two-stage simulation
# ----------------------------
def run_cobelli(t_min, p, Ix_g_min, Iu_u_min):
    u11_0 = p['u11_basal']
    u13_0 = p['m31']*u11_0/p['m13']
    u12_0 = ((p['m01']+p['m21']+p['m31'])*u11_0 - p['m13']*u13_0)/p['m12']
    F6_0  = (p['m02']+p['m12'])*u12_0 - p['m21']*u11_0
    u2p_0 = F6_0/(0.5*(1+np.tanh(p['b6']*p['c6'])))
    u1p_0 = (F6_0 + p['k12']*u2p_0)/p['k21']
    u1_0  = p['u1_basal']; u2_0 = p['u2_basal']
    y0 = np.array([u1_0,u1p_0,u2p_0,u11_0,u12_0,u13_0,u2_0],float)
    sol = odeint(lambda u,t: cobelli_rhs(u,t,p,Ix_g_min,Iu_u_min), y0, t_min)
    Ge_mgdl = sol[:,0]/(p['V1_vol']*10.0)
    return sol, Ge_mgdl

from scipy.integrate import solve_ivp

def _integrate(fun, x0, t_eval_ms, max_step_ms=0.5):
    t0, t1 = float(t_eval_ms[0]), float(t_eval_ms[-1])
    sol = solve_ivp(lambda t,y: fun(y,t), (t0, t1), x0,
                    t_eval=t_eval_ms, method="BDF",
                    rtol=1e-6, atol=1e-9, max_step=max_step_ms)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T

def run_dual_osc(t_eval_ms, t_cb_min, Ge_mM_series, p, burnin_min=10.0,
                 dt_ms_burn=0.5, max_step_ms=0.5):

    Ge_interp = lambda tms: np.interp(tms/60000.0, t_cb_min, Ge_mM_series)

    # Burn-in: integrate with small internal step, sample sparsely (e.g., 50 ms)
    Ge_basal = Ge_mM_series[0]
    Ge_const = lambda _: Ge_basal
    tb = np.arange(0.0, burnin_min*60000.0 + 50.0, 50.0)  # 50 ms output is plenty for burn-in

    x0 = np.array([Ge_basal, 120.0, 380.0, 0.6, 1.0, 0.12, 150.0, -60.0, 0.1, 0.1, 100.0, 1500.0], float)
    burn_fun = lambda y,t: dual_rhs(y,t,Ge_const,p)
    x_burn = _integrate(burn_fun, x0, tb, max_step_ms=dt_ms_burn)
    x_init = x_burn[-1,:]

    # Main run: small internal step, **coarser output grid**
    main_fun = lambda y,t: dual_rhs(y,t,Ge_interp,p)
    return _integrate(main_fun, x_init, t_eval_ms, max_step_ms=max_step_ms)



# ----------------------------
# Example: 60 min with 100 mg/kg·min IV bolus for 3 min
# ----------------------------
def Ix_glucose_bolus_gmin(t_min):
    # 70 kg -> 7 g/min for first 3 min
    return 7.0 if (0.0 <= t_min <= 3.0) else 0.0

def Iu_insulin_zero(t_min): return 0.0

if __name__ == "__main__":
    #je potřeba pak zvýšit na 120
    SIM_MIN = 20.0

    # Cobelli (unchanged)
    t_cb_min = np.linspace(0, SIM_MIN, int(SIM_MIN*20)+1)
    cb_sol, Ge_mgdl = run_cobelli(t_cb_min, params_cb, Ix_glucose_bolus_gmin, Iu_insulin_zero)
    Ge_mM = mgdl_to_mM(Ge_mgdl)

    # Dual oscillator
    dt_internal_ms = 0.5   # internal solver step ceiling
    dt_out_ms      = 10.0   # output sampling (10 ms also fine)
    t_ms_out = np.arange(0.0, SIM_MIN*60000.0 + dt_out_ms, dt_out_ms)

    do_sol = run_dual_osc(
        t_eval_ms=t_ms_out,
        t_cb_min=t_cb_min,
        Ge_mM_series=Ge_mM,
        p=params_do,
        burnin_min=10.0,
        dt_ms_burn=dt_internal_ms,
        max_step_ms=dt_internal_ms
    )

    # Plot
    t_ms_min = t_ms_out / 60000.0
    fig, axs = plt.subplots(4,1,figsize=(10,10))
    axs[0].plot(t_cb_min, Ge_mM);       axs[0].set_ylabel("Ge (mM)")
    axs[1].plot(t_ms_min, do_sol[:,7]); axs[1].set_ylabel("V (mV)")
    axs[2].plot(t_ms_min, do_sol[:,9]); axs[2].set_ylabel("[Ca2+]c (µM)")
    axs[3].plot(t_ms_min, do_sol[:,2]); axs[3].set_ylabel("FBP (µM)"); axs[3].set_xlabel("Time (min)")
    plt.tight_layout(); plt.show()