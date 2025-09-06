#!/usr/bin/env python3
"""
Glucose–Insulin Model Comparison Suite (auto-run, no CLI)

Runs all models & scenarios and produces:
- Per-model time series (G & I)
- Cross-model overlays (absolute + normalized to basal) on 0–180 min
- Phase-plane loops (G vs I)
- Return-to-basal plots (ΔG, ΔI)
- Insulin FFT spectra with dominant period annotation
- G↔I cross-correlation plots (phase lag estimate)
- Summary metrics CSV: peak, time-to-peak, AUC, half-life
- Small one-parameter sensitivity sweeps for each model (figures + CSV)

Outputs go to ./gi_suite_output (next to where you run this).
"""

from __future__ import annotations
import math, os, csv
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Dict, List
from pathlib import Path

import numpy as np
import matplotlib
# Use a non-interactive backend so this works on headless systems as well
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# Output location (relative to where you run the program)
# ======================================================================
OUTDIR = Path.cwd() / "gi_suite_output"

# ======================================================================
# Utilities
# ======================================================================
def simulate_ode(f, y0, t0, t1, dt):
    """Fixed-step RK4 integrator."""
    n_steps = int((t1 - t0) / dt) + 1
    t = np.linspace(t0, t1, n_steps)
    y = np.zeros((n_steps, len(y0)), dtype=float)
    y[0] = y0
    for k in range(n_steps - 1):
        tk = t[k]; yk = y[k]; h = dt
        k1 = np.asarray(f(tk, yk))
        k2 = np.asarray(f(tk + h/2, yk + h*k1/2))
        k3 = np.asarray(f(tk + h/2, yk + h*k2/2))
        k4 = np.asarray(f(tk + h, yk + h*k3))
        y[k+1] = yk + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t, y

def pos_part(x): 
    return x if x > 0.0 else 0.0

def ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path.as_posix(), dpi=200)
    plt.close()
    return path

# ======================================================================
# Models
# ======================================================================
def run_minimal(params=None):
    """Bergman/Cobelli-style minimal model (IVGTT-like)."""
    P = dict(
        Gb=90.0, Ib=10.0,
        Sg=0.02, p2=0.02, p3=1e-4,
        n=0.1, phi=0.5, h_g=80.0,
        t0=0.0, t1=180.0, dt=0.1,
        bolus_time=10.0, bolus_dur=0.5, bolus_area=300.0
    )
    if params: P.update(params)

    def Gin(t):
        if P["bolus_time"] <= t < P["bolus_time"] + P["bolus_dur"]:
            return P["bolus_area"]/P["bolus_dur"]
        return 0.0

    def f(t, y):
        G, X, I = y
        dG = - (P["Sg"] + X) * (G - P["Gb"]) + Gin(t)
        dX = - P["p2"] * X + P["p3"] * (I - P["Ib"])
        dI = - P["n"] * (I - P["Ib"]) + P["phi"] * pos_part(G - P["h_g"])
        return [dG, dX, dI]

    y0 = [P["Gb"], 0.0, P["Ib"]]
    t, y = simulate_ode(f, y0, P["t0"], P["t1"], P["dt"])
    return {"name":"minimal","Gb":P["Gb"],"Ib":P["Ib"],"t":t,"G":y[:,0],"I":y[:,2]}

def run_integrated(params=None):
    """Integrated (Cobelli-like) — two-phase insulin + glucagon modulation."""
    P = dict(
        Gb=90.0, Ib=10.0, G0=85.0,
        HGP0=1.5, alpha_I=0.04, alpha_Gc=0.06,
        Uii=1.0, k_util=0.01,
        n_plasma=0.15, E_pc=0.25, n_inter=0.05,
        n_gcg=0.05, beta_gcg_I=0.02, beta_gcg_G=0.03,
        k_store_fill=0.5, k_release=0.8, store_max=500.0,
        second_phase_gain=0.2,
        t0=0.0, t1=180.0, dt=0.1,
        inf_start=10.0, inf_dur=10.0, inf_rate=3.0
    )
    if params: P.update(params)

    def Ginf(t):
        if P["inf_start"] <= t < P["inf_start"] + P["inf_dur"]:
            return P["inf_rate"]
        return 0.0

    def f(t, y):
        G, Ip, Ic, Gcg, Store, RRP = y
        HGP = P["HGP0"] * (1.0 - math.tanh(P["alpha_I"] * (Ic - P["Ib"]))
                           + math.tanh(P["alpha_Gc"] * (Gcg - 50.0)))
        Util = P["Uii"] + P["k_util"] * max(Ic, 0.0) * max(G - P["Gb"], 0.0)
        dG = HGP + Ginf(t) - Util

        dStore_fill = P["k_store_fill"] * pos_part(G - P["G0"]) * (P["store_max"] - Store) - 0.2 * pos_part(Store)
        dRRP = 0.4 * Store - P["k_release"] * RRP
        secretion = 0.02 * P["k_release"] * RRP + P["second_phase_gain"] * pos_part(G - P["G0"])

        dIp = secretion - P["n_plasma"] * (Ip - P["Ib"]) - P["E_pc"] * (Ip - Ic)
        dIc = P["E_pc"] * (Ip - Ic) - P["n_inter"] * (Ic - P["Ib"])

        prod_gcg = 2.0 * (1.0 + math.tanh(P["beta_gcg_G"] * (80.0 - G))) \
                   * (1.0 + math.tanh(P["beta_gcg_I"] * (12.0 - Ic)))
        dGcg = prod_gcg - P["n_gcg"] * Gcg

        return [dG, dIp, dIc, dGcg, dStore_fill, dRRP]

    y0 = [P["Gb"], P["Ib"], P["Ib"], 50.0, 200.0, 50.0]
    t, y = simulate_ode(f, y0, P["t0"], P["t1"], P["dt"])
    return {"name":"integrated","Gb":P["Gb"],"Ib":P["Ib"],"t":t,"G":y[:,0],"I":y[:,1]}

def run_sturis(params=None):
    """Sturis ultradian — delayed insulin action → ~100–150 min oscillations."""
    P = dict(
        Ib=10.0, Gb=100.0, R_B=1.0,
        Vmax_T=3.0, Km_T=90.0, alpha_T=0.015,
        HGP_base=2.4, k_H_inhib=0.06,
        tp=75.0, tc=60.0, E=0.05,
        S_max=50.0/60.0, G_mid=100.0, s_slope=0.08,
        k_delay=0.03, Gin_const=200.0/100.0,
        t0=0.0, t1=600.0, dt=0.1
    )
    if params: P.update(params)

    def Gin(t): return P["Gin_const"]
    def R_T(G, Ic): return (P["Vmax_T"] * G / (P["Km_T"] + G)) * (1.0 + P["alpha_T"] * max(Ic - P["Ib"], 0.0))
    def R_L(x3): return P["HGP_base"] * (1.0 - math.tanh(P["k_H_inhib"] * (x3 - P["Ib"])))
    def S(G): return P["S_max"] * (1.0 + math.tanh(P["s_slope"] * (G - P["G_mid"])))

    def f(t, y):
        G, Ip, Ic, x1, x2, x3 = y
        dG  = R_L(x3) - R_T(G, Ic) - P["R_B"] + Gin(t)
        dIp = S(G) - (Ip - P["Ib"])/P["tp"] - P["E"]*(Ip - Ic)
        dIc = P["E"]*(Ip - Ic) - (Ic - P["Ib"])/P["tc"]
        dx1 = P["k_delay"]*(Ic - x1)
        dx2 = P["k_delay"]*(x1 - x2)
        dx3 = P["k_delay"]*(x2 - x3)
        return [dG, dIp, dIc, dx1, dx2, dx3]

    y0 = [P["Gb"], P["Ib"], P["Ib"], P["Ib"], P["Ib"], P["Ib"]]
    t, y = simulate_ode(f, y0, P["t0"], P["t1"], P["dt"])
    return {"name":"sturis","Gb":P["Gb"],"Ib":P["Ib"],"t":t,"G":y[:,0],"I":y[:,1]}

# ======================================================================
# Analysis helpers
# ======================================================================
def metrics(run):
    t, G, I = run["t"], run["G"], run["I"]
    dt = t[1]-t[0]
    Gb, Ib = run["Gb"], run["Ib"]
    G_peak = float(np.max(G)); I_peak = float(np.max(I))
    t_G_peak = float(t[int(np.argmax(G))]); t_I_peak = float(t[int(np.argmax(I))])
    dG = np.maximum(G-Gb, 0.0); dI = np.maximum(I-Ib, 0.0)
    AUC_G = float(np.trapz(dG, t)); AUC_I = float(np.trapz(dI, t))
    def half_life(y, yb, t):
        ypk = np.max(y); target = yb + 0.5*(ypk - yb)
        idx_start = int(np.argmax(y))
        for k in range(idx_start, len(y)):
            if y[k] <= target:
                return float(t[k] - t[idx_start])
        return float("nan")
    HL_G = half_life(G, Gb, t); HL_I = half_life(I, Ib, t)
    return dict(G_peak=G_peak, t_G_peak=t_G_peak, I_peak=I_peak, t_I_peak=t_I_peak,
                AUC_G=AUC_G, AUC_I=AUC_I, HL_G=HL_G, HL_I=HL_I)

def cross_correlation_lag(run, max_lag_min=120.0):
    t, G, I = run["t"], run["G"], run["I"]
    dt = t[1]-t[0]
    max_lag = int(max_lag_min/dt)
    g = (G - np.mean(G)); i = (I - np.mean(I))
    corr = np.correlate(g, i, mode="full")
    lags = np.arange(-len(g)+1, len(g)) * dt
    mask = (lags >= -max_lag*dt) & (lags <= max_lag*dt)
    lags = lags[mask]; corr = corr[mask]
    best = int(np.argmax(corr))
    return lags, corr, float(lags[best])

# ======================================================================
# Plotting
# ======================================================================
def plot_timeseries(run, title, fname):
    t,G,I = run["t"], run["G"], run["I"]
    plt.figure(figsize=(9,4)); plt.plot(t,G,label="Glucose (mg/dL)"); plt.plot(t,I,label="Insulin (µU/mL)")
    plt.xlabel("Time (min)"); plt.ylabel("Level"); plt.legend(); plt.title(title)
    return savefig(fname)

def overlay(runs, var, title, fname, tmax=None, normalize=False):
    plt.figure(figsize=(9,4))
    for r in runs:
        t = r["t"]; y = r[var]
        if tmax is not None:
            idx = t <= tmax; t = t[idx]; y = y[idx]
        if normalize:
            base = r["Gb"] if var=="G" else r["Ib"]
            denom = max(1e-9, abs(y.max() - base))
            y = (y - base) / denom
        plt.plot(t, y, label=r["name"])
    plt.xlabel("Time (min)")
    plt.ylabel(("Normalized " if normalize else "") + ("Glucose" if var=="G" else "Insulin"))
    plt.legend(); plt.title(title)
    return savefig(fname)

def phase_plane(runs, title, fname, tmax=None):
    plt.figure(figsize=(5.5,5.5))
    for r in runs:
        t,G,I = r["t"], r["G"], r["I"]
        if tmax is not None:
            idx = t <= tmax; G=G[idx]; I=I[idx]
        plt.plot(G,I,label=r["name"])
    plt.xlabel("Glucose (mg/dL)"); plt.ylabel("Insulin (µU/mL)"); plt.legend(); plt.title(title)
    return savefig(fname)

def return_to_basal(run, title, fname):
    Gb, Ib = run["Gb"], run["Ib"]; t = run["t"]
    dG = run["G"]-Gb; dI = run["I"]-Ib
    plt.figure(figsize=(9,4)); plt.plot(t,dG,label="G - Gb"); plt.plot(t,dI,label="I - Ib")
    plt.axhline(0.0, linestyle="--"); plt.xlabel("Time (min)"); plt.ylabel("Deviation from basal")
    plt.legend(); plt.title(title); return savefig(fname)

def fft_insulin(run, title, fname):
    t, I = run["t"], run["I"]
    dt = t[1]-t[0]
    I_d = I - np.mean(I)
    N = len(I_d)
    freqs = np.fft.rfftfreq(N, d=dt)
    mag = np.abs(np.fft.rfft(I_d))/N
    if len(mag)>1:
        idx = np.argmax(mag[1:])+1; f = freqs[idx]; per = (1.0/f) if f>0 else float("inf")
        title = f"{title} (dominant period ≈ {per:.0f} min)"
    plt.figure(figsize=(9,4)); plt.plot(freqs, mag, label="|FFT(I)|")
    plt.xlabel("Frequency (cycles/min)"); plt.ylabel("Magnitude")
    plt.title(title); plt.legend(); return savefig(fname)

def xcorr_plot(run, title, fname):
    lags, corr, best = cross_correlation_lag(run)
    plt.figure(figsize=(9,4)); plt.plot(lags, corr, label="xcorr(G, I)")
    plt.axvline(best, linestyle="--", label=f"peak lag ≈ {best:.1f} min")
    plt.xlabel("Lag (min, G leads +)"); plt.ylabel("Correlation (unnormalized)")
    plt.title(title); plt.legend(); return savefig(fname)

# ======================================================================
# Sensitivity sweeps
# ======================================================================
def sensitivity_minimal():
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    for f in factors:
        r = run_minimal({"n": 0.1*f})
        m = metrics(r); results.append((f, m["AUC_G"], m["AUC_I"]))
    plt.figure(figsize=(7,4))
    x = [f for f,_,_ in results]; aucg = [a for _,a,_ in results]; auch = [b for _,_,b in results]
    plt.plot(x, aucg, label="AUC_G vs insulin clearance n")
    plt.plot(x, auch, label="AUC_I vs insulin clearance n")
    plt.xlabel("Scaling of n"); plt.ylabel("AUC"); plt.legend()
    return savefig("S1_minimal_sensitivity.png"), results

def sensitivity_integrated():
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    for f in factors:
        r = run_integrated({"second_phase_gain": 0.2*f})
        m = metrics(r); results.append((f, m["I_peak"], m["AUC_I"]))
    plt.figure(figsize=(7,4))
    x = [f for f,_,_ in results]; y1 = [a for _,a,_ in results]; y2 = [b for _,_,b in results]
    plt.plot(x, y1, label="I_peak vs second_phase_gain")
    plt.plot(x, y2, label="AUC_I vs second_phase_gain")
    plt.xlabel("Scaling"); plt.ylabel("Value"); plt.legend()
    return savefig("S2_integrated_sensitivity.png"), results

def sensitivity_sturis():
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    for f in factors:
        r = run_sturis({"k_delay": 0.03*f})
        # dominant period from FFT
        t, I = r["t"], r["I"]; dt = t[1]-t[0]
        I_d = I - np.mean(I); freqs = np.fft.rfftfreq(len(I_d), d=dt); mag = np.abs(np.fft.rfft(I_d))
        idx = np.argmax(mag[1:])+1; fdom = freqs[idx]; per = (1.0/fdom) if fdom>0 else float("nan")
        results.append((f, per))
    plt.figure(figsize=(7,4))
    x = [f for f,_ in results]; y = [p for _,p in results]
    plt.plot(x, y, label="Dominant period vs k_delay")
    plt.xlabel("Scaling of k_delay"); plt.ylabel("Period (min)"); plt.legend()
    return savefig("S3_sturis_sensitivity.png"), results

# ======================================================================
# Orchestrator
# ======================================================================
def main():
    ensure_outdir()

    # Baseline runs
    r_min = run_minimal()
    r_int = run_integrated()
    r_stu = run_sturis()
    runs_all = [r_min, r_int, r_stu]

    # A: Per-model time series
    plot_timeseries(r_min, "Minimal — IV bolus", "A1_minimal_timeseries.png")
    plot_timeseries(r_int, "Integrated — 10-min infusion, two-phase insulin", "A2_integrated_timeseries.png")
    plot_timeseries(r_stu, "Sturis — constant drive, ultradian oscillations", "A3_sturis_timeseries.png")

    # B: Cross-model overlays (0–180 min)
    overlay(runs_all, "G", "Glucose overlays (trimmed to 0–180)", "B1_overlay_G_0_180.png", tmax=180)
    overlay(runs_all, "I", "Insulin overlays (trimmed to 0–180)", "B2_overlay_I_0_180.png", tmax=180)
    overlay(runs_all, "G", "Glucose overlays — normalized", "B3_overlay_G_norm_0_180.png", tmax=180, normalize=True)
    overlay(runs_all, "I", "Insulin overlays — normalized", "B4_overlay_I_norm_0_180.png", tmax=180, normalize=True)

    # C: Phase-plane
    phase_plane(runs_all, "Phase-plane G vs I (0–180)", "C1_phase_plane_0_180.png", tmax=180)
    phase_plane([r_stu], "Phase-plane G vs I (Sturis full horizon)", "C2_phase_plane_sturis.png")

    # D: Return-to-basal
    return_to_basal(r_min, "Return-to-basal — Minimal", "D1_min_return.png")
    return_to_basal(r_int, "Return-to-basal — Integrated", "D2_int_return.png")
    return_to_basal(r_stu, "Return-to-basal — Sturis", "D3_stu_return.png")

    # E: FFT of insulin
    fft_insulin(r_min, "Insulin FFT — Minimal", "E1_fft_min.png")
    fft_insulin(r_int, "Insulin FFT — Integrated", "E2_fft_int.png")
    fft_insulin(r_stu, "Insulin FFT — Sturis", "E3_fft_stu.png")

    # F: Cross-correlation lag
    xcorr_plot(r_min, "G↔I cross-corr — Minimal", "F1_xcorr_min.png")
    xcorr_plot(r_int, "G↔I cross-corr — Integrated", "F2_xcorr_int.png")
    xcorr_plot(r_stu, "G↔I cross-corr — Sturis", "F3_xcorr_stu.png")

    # G: Summary metrics CSV
    with (OUTDIR / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","G_peak","t_G_peak","I_peak","t_I_peak","AUC_G","AUC_I","HL_G","HL_I"])
        for r in runs_all:
            m = metrics(r)
            w.writerow([r["name"], m["G_peak"], m["t_G_peak"], m["I_peak"], m["t_I_peak"], m["AUC_G"], m["AUC_I"], m["HL_G"], m["HL_I"]])

    # H: Sensitivity sweeps
    _, res1 = sensitivity_minimal()
    _, res2 = sensitivity_integrated()
    _, res3 = sensitivity_sturis()

    # Save sweeps as CSVs
    for name, results, header in [
        ("sensitivity_minimal.csv", res1, ["scale_n","AUC_G","AUC_I"]),
        ("sensitivity_integrated.csv", res2, ["scale_second_phase_gain","I_peak","AUC_I"]),
        ("sensitivity_sturis.csv", res3, ["scale_k_delay","dominant_period_min"]),
    ]:
        with (OUTDIR / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(results)

    print(f"All figures and CSVs written to: {OUTDIR}")

if __name__ == "__main__":
    main()
