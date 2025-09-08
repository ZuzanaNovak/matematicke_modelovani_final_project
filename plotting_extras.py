import os, numpy as np, matplotlib.pyplot as plt
from utils import ensure_dir

def stacked_fluxes(T, f, outdir, tag):
    ensure_dir(outdir)
    plt.figure(figsize=(8,5))
    up = np.clip(f["F1"], 0, None)
    down = f["F2"] + f["F3"] + f["F4"] + f["F5"]
    plt.stackplot(T, up, labels=["Hepatic production (F1)"])
    plt.stackplot(T, -np.vstack([f["F2"], f["F3"], f["F4"], f["F5"]]),
                  labels=["Hepatic uptake (F2)","Renal (F3)","Peripheral ID (F4)","CNS/RBC IND (F5)"])
    plt.plot(T, f["NHGB"], lw=2, label="NHGB")
    plt.axhline(0,color='k',lw=0.8)
    plt.legend(loc="upper right"); plt.xlabel("time [min]"); plt.ylabel("mg/min")
    plt.title(f"Stacked fluxes — {tag}"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_stacked_fluxes.png"), dpi=200); plt.close()

def phase_plane(T, x, y, xlabel, ylabel, outpath, color_by_time=True):
    c = T if color_by_time else None
    plt.figure()
    sc = plt.scatter(x, y, c=c, s=10, cmap="viridis")
    if color_by_time: plt.colorbar(sc, label="time [min]")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"Phase-plane: {ylabel} vs {xlabel}")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def secretion_panels(T, s, y, outdir, tag):
    plt.figure(figsize=(8,5))
    plt.plot(T, s["W"],  label="β synthesis W (µU/min)")
    plt.plot(T, s["F6"], label="Insulin secretion F6 (µU/min)")
    plt.xlabel("time [min]"); plt.ylabel("µU/min"); plt.title(f"β-cell fluxes — {tag}")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_beta_fluxes.png"), dpi=200); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(T, s["F7"], label="Glucagon secretion F7 (pg/min)")
    plt.xlabel("time [min]"); plt.ylabel("pg/min"); plt.title(f"α-cell flux — {tag}")
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_alpha_flux.png"), dpi=200); plt.close()


def sensitivity_heatmap(X, Y, Z, xlab, ylab, title, outpath):
    plt.figure(figsize=(6,5))
    im = plt.imshow(Z, origin='lower', aspect='auto',
                    extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.colorbar(im, label="Glucose @120 min [mg/dl]")
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()
