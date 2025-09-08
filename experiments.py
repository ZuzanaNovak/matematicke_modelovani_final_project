import numpy as np
from utils import ensure_dir
from scenarios import build_control, build_t2d
from state import initial_state
from model import rhs, integrate
import processes as pr
from analytics import series_dict
from plotting_extras import (
    stacked_fluxes, phase_plane, secretion_panels, sensitivity_heatmap
)

OUT = ensure_dir("results_extras")

def simulate(P, t_end=180.0, dt=0.05):
    y0 = initial_state(P)
    T, Y = integrate(rhs, 0.0, t_end, dt, y0, P)
    y, f, s = series_dict(T, Y, P, pr)
    return T, y, f, s

def panels_extras(tag, T, y, f, s):
    d = ensure_dir(f"{OUT}/{tag}")
    # 1) Stacked hepatic/peripheral fluxes
    stacked_fluxes(T, f, d, tag)
    # 2) Phase-planes
    phase_plane(T, y["y1"], f["NHGB"],
                "Glucose [mg/dl]", "NHGB [mg/min]", f"{d}/{tag}_phase_y1_vs_NHGB.png")
    phase_plane(T, y["y4"], f["F4"],
                "Interstitial insulin [µU/ml]", "Peripheral ID (F4) [mg/min]",
                f"{d}/{tag}_phase_y4_vs_F4.png")
    phase_plane(T, y["y3"], f["F1"],
                "Portal insulin [µU/ml]", "Hepatic production (F1) [mg/min]",
                f"{d}/{tag}_phase_y3_vs_F1.png")
    # 3) α/β-cell flux panels
    secretion_panels(T, s, y, d, tag)

def sensitivity_grid(tag, build_P, xs, ys, xsetter, ysetter, t_eval=120.0):
    """
    Make a 2D sensitivity heatmap of glucose at t_eval.
    xs, ys: arrays of parameter values
    xsetter, ysetter: callables (P, value) -> None to set param
    """
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = np.zeros_like(X, dtype=float)

    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            P = build_P()
            xsetter(P, xv); ysetter(P, yv)
            T, y, f, s = simulate(P)
            # value at closest time to t_eval
            idx = np.argmin(np.abs(T - t_eval))
            Z[j, i] = y["y1"][idx]

    sensitivity_heatmap(
        X, Y, Z,
        xlab=xsetter.__name__, ylab=ysetter.__name__,
        title=f"Sensitivity: G({t_eval} min)",
        outpath=f"{OUT}/{tag}_sens_heatmap.png"
    )

if __name__ == "__main__":
    # ---- CONTROL ----
    P_ctrl = build_control()
    T, y, f, s = simulate(P_ctrl)
    panels_extras("control", T, y, f, s)

    # Sensitivity (control): hepatic uptake scale (a21) × peripheral sensitivity (b33)
    xs = np.linspace(0.25e-3, 0.75e-3, 13)   # a21
    ys = np.linspace(0.02, 0.05, 13)         # b33
    def set_a21(P, v): setattr(P, "a21", float(v))
    def set_b33(P, v): setattr(P, "b33", float(v))
    set_a21.__name__ = "a21 (hepatic uptake scale)"
    set_b33.__name__ = "b33 (peripheral sensitivity)"
    sensitivity_grid("control", build_control, xs, ys, set_a21, set_b33)

    # ---- T2D ----
    P_t2d = build_t2d()
    T, y, f, s = simulate(P_t2d)
    panels_extras("t2d", T, y, f, s)

    # Sensitivity (t2d): hepatic production level (a11) × glucagon inhibition shift (c71)
    xs = np.linspace(5.0, 7.5, 13)       # a11
    ys = np.linspace(110.0, 130.0, 13)   # c71
    def set_a11(P, v): setattr(P, "a11", float(v))
    def set_c71(P, v): setattr(P, "c71", float(v))
    set_a11.__name__ = "a11 (hepatic production scale)"
    set_c71.__name__ = "c71 (insulin inhibition shift)"
    sensitivity_grid("t2d", build_t2d, xs, ys, set_a11, set_c71)

    print("Extra analyses saved to ./results_extras")
