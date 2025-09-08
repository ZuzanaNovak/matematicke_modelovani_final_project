import numpy as np
from typing import Dict

def auc_trapz(T, y):
    return float(np.trapz(y, T))

def time_of_peak(T, y):
    i = int(np.argmax(y))
    return float(T[i]), float(y[i])

def time_of_min(T, y):
    i = int(np.argmin(y))
    return float(T[i]), float(y[i])

def series_dict(T, Y, P, processes):
    """
    Build all useful derived series in one go.
    Returns dicts: y (concentrations & pools), f (fluxes), s (secretions)
    """
    x1,u1p,u2p,u11,u12,u13,u2 = Y.T
    y = {
        "y1": x1/P.V1*100.0,
        "y2": u11/P.V11,
        "y3": u12/P.V12,
        "y4": u13/P.V13,
        "y5": u2/P.V2,
        "u1p": u1p,
        "u2p": u2p,
    }

    # fluxes
    F1 = np.array([processes.scale_bw(processes.F1_LiverProduction_perkg(x,uu12,uu2,P), P)
                   for x,uu12,uu2 in zip(x1,u12,u2)])
    F2 = np.array([processes.scale_bw(processes.F2_LiverUptake_perkg(x,uu12,P), P)
                   for x,uu12 in zip(x1,u12)])
    F3 = np.array([processes.scale_bw(processes.F3_RenalExcretion_perkg(x,P), P)
                   for x in x1])
    F4 = np.array([processes.scale_bw(processes.F4_PeripheralID_perkg(x,uu13,P), P)
                   for x,uu13 in zip(x1,u13)])
    F5 = np.array([processes.scale_bw(processes.F5_PeripheralInd_perkg(x,P), P)
                   for x in x1])
    NHGB = F1 - F2
    f = {"F1":F1, "F2":F2, "F3":F3, "F4":F4, "F5":F5, "NHGB":NHGB}

    # secretions (µU/min, pg/min)
    W  = np.array([processes.W_insulin_synthesis(x,P) for x in x1])
    F6 = np.array([processes.F6_secretion(x,uu2p,P)  for x,uu2p in zip(x1,u2p)])
    F7 = np.array([processes.F7_glucagon_secretion(x,uu13,P) for x,uu13 in zip(x1,u13)])
    s = {"W": W, "F6": F6, "F7": F7}
    return y, f, s

def quick_metrics(T, y: Dict[str, np.ndarray], f: Dict[str, np.ndarray]):
    """A few handy metrics for tables/legends."""
    g_peak_t, g_peak = time_of_peak(T, y["y1"])
    i_peak_t, i_peak = time_of_peak(T, y["y2"])
    nhgb_min_t, nhgb_min = time_of_min(T, f["NHGB"])
    return {
        "AUC_glucose": auc_trapz(T, y["y1"]),
        "AUC_insulin": auc_trapz(T, y["y2"]),
        "AUC_glucagon": auc_trapz(T, y["y5"]),
        "Peak_glucose": g_peak, "t_peak_glucose": g_peak_t,
        "Peak_insulin": i_peak, "t_peak_insulin": i_peak_t,
        "NHGB_min": nhgb_min, "t_NHGB_min": nhgb_min_t,
        "G120": float(y["y1"][np.argmin(np.abs(T-120))]),
        "I120": float(y["y2"][np.argmin(np.abs(T-120))]),
    }
