import os
import numpy as np
import matplotlib.pyplot as plt
from utils import ensure_dir

def plot_panels(T, y_dict, f_dict, outdir, tag):
    ensure_dir(outdir)

    # 1) Glucose
    plt.figure(); plt.plot(T, y_dict["y1"])
    plt.xlabel("Time [min]"); plt.ylabel("Glucose [mg/dl]")
    plt.title(f"Glucose — {tag}"); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_01_glucose.png"), dpi=200); plt.close()

    # 2) Insulin 
    fig,ax = plt.subplots(3,1,figsize=(7,8),sharex=True)
    ax[0].plot(T, y_dict["y2"]); ax[0].set_ylabel("plasma [µU/ml]")
    ax[1].plot(T, y_dict["y3"]); ax[1].set_ylabel("liver/portal [µU/ml]")
    ax[2].plot(T, y_dict["y4"]); ax[2].set_ylabel("interstitial [µU/ml]"); ax[2].set_xlabel("Time [min]")
    for a in ax: a.grid(True,alpha=0.3)
    fig.suptitle(f"Insulin — {tag}")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(outdir, f"{tag}_02_insulin_concs.png"), dpi=200); plt.close(fig)

    # 3) Glucagon
    plt.figure(); plt.plot(T, y_dict["y5"])
    plt.xlabel("Time [min]"); plt.ylabel("Glucagon [pg/ml]")
    plt.title(f"Glucagon — {tag}"); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_03_glucagon.png"), dpi=200); plt.close()

    # 4) Hepatic
    F1,F2,NHGB = f_dict["F1"], f_dict["F2"], f_dict["NHGB"]
    plt.figure()
    plt.plot(T,F1,label="F1 production")
    plt.plot(T,F2,label="F2 hepatic uptake")
    plt.plot(T,NHGB,label="NHGB = F1 - F2")
    plt.xlabel("Time [min]"); plt.ylabel("mg/min"); plt.title(f"Liver — {tag}")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_04_hepatic.png"), dpi=200); plt.close()

    # 5) Peripheral & Renal
    plt.figure()
    plt.plot(T, f_dict["F4"], label="F4 perif. (ID)")
    plt.plot(T, f_dict["F5"], label="F5 CNS/RBC (IND)")
    plt.plot(T, f_dict["F3"], label="F3 renal")
    plt.xlabel("Time [min]"); plt.ylabel("mg/min")
    plt.title(f"Peripheral & Renal — {tag}")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_05_periph_renal.png"), dpi=200); plt.close()

    # 6) Pools
    plt.figure()
    plt.plot(T, y_dict["u1p"], label="u1p (stored)")
    plt.plot(T, y_dict["u2p"], label="u2p (prompt)")
    plt.xlabel("Time [min]"); plt.ylabel("µU"); plt.legend()
    plt.title(f"Pancreatic Pools — {tag}")
    plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_06_pools.png"), dpi=200); plt.close()

def plot_overlays(Tc, series_c, Td, series_d, outdir):
    """series_* are dicts with keys: y1, y2, y5, NHGB"""
    ensure_dir(outdir)

    plt.figure(); plt.plot(Tc,series_c["y1"],label="Control")
    plt.plot(Td,series_d["y1"],label="T2D")
    plt.xlabel("Time [min]"); plt.ylabel("glucose [mg/dl]")
    plt.title("Glucose — Control vs. T2D"); plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_glucose.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(Tc,series_c["y2"],label="Control")
    plt.plot(Td,series_d["y2"],label="T2D")
    plt.xlabel("Time [min]"); plt.ylabel("insulin [µU/ml]")
    plt.title("Plazmatic insulin — Control vs. T2D"); plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_insulin.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(Tc,series_c["y5"],label="Control")
    plt.plot(Td,series_d["y5"],label="T2D")
    plt.xlabel("Time [min]"); plt.ylabel("glukagon [pg/ml]")
    plt.title("Glukagon — Control vs. T2D"); plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_glucagon.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(Tc,series_c["NHGB"],label="Control")
    plt.plot(Td,series_d["NHGB"],label="T2D")
    plt.xlabel("Time [min]"); plt.ylabel("NHGB [mg/min]")
    plt.title("NHGB — Control vs. T2D"); plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_NHGB.png"), dpi=200); plt.close()
