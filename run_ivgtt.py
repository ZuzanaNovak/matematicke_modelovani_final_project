import numpy as np
from utils import ensure_dir, value_at
from params import set_volumes
from scenarios import build_control, build_t2d
from state import initial_state
from model import rhs, integrate
from processes import (
    scale_bw, F1_LiverProduction_perkg, F2_LiverUptake_perkg, F3_RenalExcretion_perkg,
    F4_PeripheralID_perkg, F5_PeripheralInd_perkg
)
from plotting import plot_panels, plot_overlays

def simulate(P, tag, outdir):
    ensure_dir(outdir)
    y0 = initial_state(P)
    T, Y = integrate(rhs, 0.0, 180.0, 0.05, y0, P)

    x1,u1p,u2p,u11,u12,u13,u2 = Y.T
    y1 = x1/P.V1*100.0; y2 = u11/P.V11; y3 = u12/P.V12; y4 = u13/P.V13; y5 = u2/P.V2

    # fluxes
    F1 = np.array([scale_bw(F1_LiverProduction_perkg(x,uu12,uu2,P), P) for x,uu12,uu2 in zip(x1,u12,u2)])
    F2 = np.array([scale_bw(F2_LiverUptake_perkg(x,uu12,P), P)        for x,uu12 in zip(x1,u12)])
    F3 = np.array([scale_bw(F3_RenalExcretion_perkg(x,P), P)          for x in x1])
    F4 = np.array([scale_bw(F4_PeripheralID_perkg(x,uu13,P), P)       for x,uu13 in zip(x1,u13)])
    F5 = np.array([scale_bw(F5_PeripheralInd_perkg(x,P), P)           for x in x1])
    NHGB = F1 - F2

    y_dict = {"y1":y1,"y2":y2,"y3":y3,"y4":y4,"y5":y5,"u1p":u1p,"u2p":u2p}
    f_dict = {"F1":F1,"F2":F2,"F3":F3,"F4":F4,"F5":F5,"NHGB":NHGB}

    plot_panels(T, y_dict, f_dict, outdir, tag)

    summary = {
        "glucose_peak": float(np.max(y1)),
        "glucose_60": value_at(60,T,y1),
        "glucose_120": value_at(120,T,y1),
        "insulin_peak": float(np.max(y2)),
        "insulin_120": value_at(120,T,y2),
        "glucagon_nadir": float(np.min(y5)),
        "glucagon_180": value_at(180,T,y5),
        "NHGB_min": float(np.min(NHGB)),
        "NHGB_tmin": float(T[np.argmin(NHGB)]),
        "NHGB_60": value_at(60,T,NHGB),
        "NHGB_120": value_at(120,T,NHGB),
    }
    return T, y_dict, f_dict, summary

if __name__ == "__main__":
    base = ensure_dir("results_modular")
    d_control = ensure_dir(f"{base}/control")
    d_t2d     = ensure_dir(f"{base}/t2d")
    d_ol      = ensure_dir(f"{base}/overlays")

    # Control
    P_ctrl = build_control()
    Tc, yc, fc, Sc = simulate(P_ctrl, "control", d_control)

    # T2D
    P_t2d = build_t2d()
    Td, yd, fd, Sd = simulate(P_t2d, "t2d", d_t2d)

    # Overlays
    plot_overlays(
        Tc, {"y1":yc["y1"], "y2":yc["y2"], "y5":yc["y5"], "NHGB":fc["NHGB"]},
        Td, {"y1":yd["y1"], "y2":yd["y2"], "y5":yd["y5"], "NHGB":fd["NHGB"]},
        d_ol
    )

