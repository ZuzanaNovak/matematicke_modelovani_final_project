# run_scenarios.py
# Driver, který volá funkcionality z main.py a simuluje Control vs. T2D-like.
# Obrázky a souhrny uloží do ./results_scenarios/{control,t2d,overlays}

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
import main as m  # <— MUSÍ být ve stejném adresáři, náš původní (vylepšený) model

# ---------- utility ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def maybe(attr, default=None):
    return getattr(m.P, attr, default)

def value_at(tmin, T, series):
    return float(series[np.argmin(np.abs(T - tmin))])

def solve_glucagon_gain_if_available(P):
    """Použij m.solve_glucagon_gain, pokud existuje, jinak spočítej lokálně."""
    if hasattr(m, "solve_glucagon_gain"):
        P.a7 = m.solve_glucagon_gain(P)
        return P
    # fallback výpočet: F7_bas = h01 * y5_bas * V2
    b71 = getattr(P, "b71", 0.005); c71 = getattr(P, "c71", 110.0)
    b72 = getattr(P, "b72", 0.070); c72 = getattr(P, "c72", 48.0)
    H7b = 0.5*(1 - np.tanh(b71*(0 + c71)))
    M7b = 0.5*(1 - np.tanh(b72*(0 + c72)))
    target = P.h01 * P.y5_bas * P.V2
    P.a7 = target / max(1e-12, H7b*M7b)
    return P

# ---------- scenario configs (jen úpravy parametrů; rovnice/řešič bere main.py) ----------
def build_control():
    # Vezmeme default Params z main a aplikujeme recommended „paperfit_v3“ styl, pokud tam ty položky jsou
    P = m.set_volumes(m.Params())
    # Pokud máš v mainu cohort basals, nech klidně default (nebo odkomentuj):
    if hasattr(P, "y1_bas"): P.y1_bas = 77.0
    if hasattr(P, "y2_bas"): P.y2_bas = 7.0
    if hasattr(P, "y3_bas"): P.y3_bas = 7.0
    if hasattr(P, "y4_bas"): P.y4_bas = 7.0
    if hasattr(P, "y5_bas"): P.y5_bas = 120.0

    # jemné doladění (jen když v Params existuje)
    for k, v in dict(
        k01=0.03, a6=2.4, c6=-10.0,
        m01=0.110, m02=0.170, m13=0.018, m31=0.060,
        a11=5.5, b11=1.60, c11=-1.10, c13=26.0,
        a21=0.5e-3, b21=0.10, c21=10.0, a221=60.0, a222=1.00, b22=0.090, c22=-90.0,
        b71=0.0053, c71=115.0, b72=0.070, c72=52.0,
        a31=1.0e-2, b33=0.035, c33=-32.0,
    ).items():
        if hasattr(P, k): setattr(P, k, v)

    P = solve_glucagon_gain_if_available(P)
    return P

def build_t2d():
    # T2D-like: hyperglykémie, hypoinsulinémie, sekreční defekt, inzulinová rezistence (periferní i jaterní), vyšší glukagon
    P = m.set_volumes(m.Params())

    # bazály
    if hasattr(P, "y1_bas"): P.y1_bas = 115.0   # vyšší bazální glukóza
    if hasattr(P, "y2_bas"): P.y2_bas = 5.0     # nižší inzulin
    if hasattr(P, "y3_bas"): P.y3_bas = 5.0
    if hasattr(P, "y4_bas"): P.y4_bas = 5.0
    if hasattr(P, "y5_bas"): P.y5_bas = 150.0   # vyšší glukagon

    # β-buňka: horší první fáze a celkově menší sekrece
    for k, v in dict(k01=0.02, a6=1.4, c6=-6.0).items():
        if hasattr(P, k): setattr(P, k, v)

    # inzulinová rezistence – periferie (F4): slabší citlivost na y4 a glukózu
    # (menší sklon a posun doprava)
    for k, v in dict(b33=0.022, c33=-40.0, b32=0.022, c32=-10.0, a31=8.0e-3).items():
        if hasattr(P, k): setattr(P, k, v)

    # inzulinová rezistence – játra (F2): menší a21, a221; posun tak, aby uptake nebyl tak silný
    for k, v in dict(a21=0.35e-3, a221=40.0, a222=0.6, b21=0.06, c21=25.0, b22=0.06, c22=-70.0).items():
        if hasattr(P, k): setattr(P, k, v)

    # vyšší produkce jater (F1): lehce větší a11, slabší inhibice glukózou (menší b13 / větší c13)
    for k, v in dict(a11=6.2, b11=1.50, c11=-0.9, b13=0.022, c13=24.0).items():
        if hasattr(P, k): setattr(P, k, v)

    # glukagon: silnější, hůře inhibovaný
    for k, v in dict(b71=0.0045, c71=120.0, b72=0.060, c72=55.0).items():
        if hasattr(P, k): setattr(P, k, v)

    # kinetika inzulinu: trochu rychlejší clearance + pomalejší přestup do intersticia → nižší periferní účinek
    for k, v in dict(m01=0.125, m02=0.185, m31=0.045, m13=0.022).items():
        if hasattr(P, k): setattr(P, k, v)

    P = solve_glucagon_gain_if_available(P)
    return P

# ---------- simulation ----------
def simulate(P, tag, outdir):
    ensure_dir(outdir)

    # počáteční stav
    y0 = m.initial_state(P)

    # časování – vezmi hodnoty z main, pokud jsou definované, jinak default
    t0, tf, dt = 0.0, 180.0, 0.05
    # integrace
    T, Y = m.integrate(m.rhs, t0, tf, dt, y0, P)

    # koncentrace
    x1,u1p,u2p,u11,u12,u13,u2 = Y.T
    y1 = x1 / P.V1 * 100.0
    y2 = u11 / P.V11
    y3 = u12 / P.V12
    y4 = u13 / P.V13
    y5 = u2  / P.V2

    # fluxy (volat z mainu pokud je má; jinak jen NHGB přes rhs v sobě)
    def has(name): return hasattr(m, name)

    if has("F1_LiverProduction_perkg") and has("F2_LiverUptake_perkg") and has("F3_RenalExcretion_perkg") and has("F4_PeripheralID_perkg") and has("F5_PeripheralInd_perkg") and has("scale_bw"):
        F1 = np.array([m.scale_bw(m.F1_LiverProduction_perkg(x,uu12,uu2,P), P) for x,uu12,uu2 in zip(x1,u12,u2)])
        F2 = np.array([m.scale_bw(m.F2_LiverUptake_perkg(x,uu12,P), P)        for x,uu12 in zip(x1,u12)])
        F3 = np.array([m.scale_bw(m.F3_RenalExcretion_perkg(x,P), P)          for x in x1])
        F4 = np.array([m.scale_bw(m.F4_PeripheralID_perkg(x,uu13,P), P)       for x,uu13 in zip(x1,u13)])
        F5 = np.array([m.scale_bw(m.F5_PeripheralInd_perkg(x,P), P)           for x in x1])
        NHGB = F1 - F2
    else:
        # nouzově NHGB odhadneme ze stavů (není ideální), raději měj F* v main.py
        F1 = F2 = F3 = F4 = F5 = np.full_like(y1, np.nan)
        # NHGB v main.rhs je v dx1 spolu s F3,F4,F5,Z1, takže bez F* sem ho čistě nevytáhneme.

        # fallback: spočti znovu rhs a vytáhni komponenty když máš F* (jinak ne)
        NHGB = np.full_like(y1, np.nan)

    # panely
    plt.figure(); plt.plot(T,y1); plt.xlabel("čas [min]"); plt.ylabel("glukóza [mg/dl]")
    plt.title(f"Glukóza — {tag}"); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_01_glucose.png"), dpi=200); plt.close()

    fig,ax = plt.subplots(3,1,figsize=(7,8),sharex=True)
    ax[0].plot(T,y2); ax[0].set_ylabel("plazma [µU/ml]")
    ax[1].plot(T,y3); ax[1].set_ylabel("játra/portál [µU/ml]")
    ax[2].plot(T,y4); ax[2].set_ylabel("intersticium [µU/ml]"); ax[2].set_xlabel("čas [min]")
    for a in ax: a.grid(True,alpha=0.3)
    fig.suptitle(f"Inzulin — {tag}")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(outdir, f"{tag}_02_insulin_concs.png"), dpi=200); plt.close(fig)

    plt.figure(); plt.plot(T,y5); plt.xlabel("čas [min]"); plt.ylabel("glukagon [pg/ml]")
    plt.title(f"Glukagon — {tag}"); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_03_glucagon.png"), dpi=200); plt.close()

    if not np.isnan(F1).all():
        plt.figure()
        plt.plot(T,F1,label="F1 produkce")
        plt.plot(T,F2,label="F2 jaterní uptake")
        plt.plot(T,NHGB,label="NHGB = F1 - F2")
        plt.xlabel("čas [min]"); plt.ylabel("mg/min"); plt.title(f"Játra — {tag}")
        plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_04_hepatic.png"), dpi=200); plt.close()

        plt.figure()
        plt.plot(T,F4,label="F4 perif. inzulin-dep.")
        plt.plot(T,F5,label="F5 CNS/RBC (nezávislé)")
        plt.plot(T,F3,label="F3 ledviny")
        plt.xlabel("čas [min]"); plt.ylabel("mg/min"); plt.title(f"Periferie & ledviny — {tag}")
        plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_05_periph_renal.png"), dpi=200); plt.close()

    plt.figure()
    plt.plot(T, m.P.y2_bas*np.ones_like(T), '--', alpha=0.3, label="bazál (ref.)")
    plt.plot(T,y2,label="plazma")
    plt.plot(T,y4,label="intersticium")
    plt.xlabel("čas [min]"); plt.ylabel("inzulin [µU/ml]")
    plt.title(f"Plazma vs. intersticium — {tag}")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_06_y2_vs_y4.png"), dpi=200); plt.close()

    # souhrn
    summary = {
        "glucose_peak": float(np.max(y1)),
        "glucose_60": value_at(60,T,y1),
        "glucose_120": value_at(120,T,y1),
        "insulin_peak": float(np.max(y2)),
        "insulin_120": value_at(120,T,y2),
        "glucagon_nadir": float(np.min(y5)),
        "glucagon_180": value_at(180,T,y5),
    }
    if not np.isnan(F1).all():
        summary.update({
            "NHGB_min": float(np.min(NHGB)),
            "NHGB_tmin": float(T[np.argmin(NHGB)]),
            "NHGB_60": value_at(60,T,NHGB),
            "NHGB_120": value_at(120,T,NHGB),
        })
    return T, y1, y2, y5, locals(), summary

# ---------- main ----------
if __name__ == "__main__":
    base = ensure_dir("results_scenarios")
    d_control = ensure_dir(os.path.join(base, "control"))
    d_t2d     = ensure_dir(os.path.join(base, "t2d"))
    d_ol      = ensure_dir(os.path.join(base, "overlays"))

    # Control
    P_ctrl = build_control()
    T_c, g_c, i_c, gg_c, env_c, S_c = simulate(P_ctrl, "control", d_control)

    # T2D
    P_t2d = build_t2d()
    T_d, g_d, i_d, gg_d, env_d, S_d = simulate(P_t2d, "t2d", d_t2d)

    # Overlays (glukóza, inzulin, glukagon, NHGB pokud máme)
    plt.figure(); plt.plot(T_c,g_c,label="Control"); plt.plot(T_d,g_d,label="T2D")
    plt.xlabel("čas [min]"); plt.ylabel("glukóza [mg/dl]"); plt.title("Glukóza — Control vs. T2D")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(d_ol, "overlay_glucose.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(T_c,i_c,label="Control"); plt.plot(T_d,i_d,label="T2D")
    plt.xlabel("čas [min]"); plt.ylabel("inzulin [µU/ml]"); plt.title("Plazmatický inzulin — Control vs. T2D")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(d_ol, "overlay_insulin.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(T_c,gg_c,label="Control"); plt.plot(T_d,gg_d,label="T2D")
    plt.xlabel("čas [min]"); plt.ylabel("glukagon [pg/ml]"); plt.title("Glukagon — Control vs. T2D")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(d_ol, "overlay_glucagon.png"), dpi=200); plt.close()

    # NHGB overlay jen pokud existují fluxy
    F1_c = env_c.get("F1", None); F2_c = env_c.get("F2", None)
    F1_d = env_d.get("F1", None); F2_d = env_d.get("F2", None)
    if F1_c is not None and not np.isnan(F1_c).all() and F1_d is not None and not np.isnan(F1_d).all():
        NHGB_c = env_c["F1"] - env_c["F2"]
        NHGB_d = env_d["F1"] - env_d["F2"]
        plt.figure(); plt.plot(T_c,NHGB_c,label="Control"); plt.plot(T_d,NHGB_d,label="T2D")
        plt.xlabel("čas [min]"); plt.ylabel("NHGB [mg/min]"); plt.title("NHGB — Control vs. T2D")
        plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(d_ol, "overlay_NHGB.png"), dpi=200); plt.close()

    # textové souhrny do konzole
    print("== SUMMARY: CONTROL =="); [print(f"{k}: {v}") for k,v in S_c.items()]
    print("\n== SUMMARY: T2D ==");    [print(f"{k}: {v}") for k,v in S_d.items()]
    print(f"\nVýstupy: {base}")
