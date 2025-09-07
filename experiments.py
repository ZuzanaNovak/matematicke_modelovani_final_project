# experiments.py
import numpy as np
from utils import ensure_dir, value_at
from params import set_volumes
from scenarios import build_control, build_t2d
from state import initial_state
from model import rhs, integrate
import processes as pr
from analytics import series_dict, quick_metrics
from plotting_extras import (
    stacked_fluxes, phase_plane, secretion_panels, ratios_and_times,
    dose_response_plot, sensitivity_heatmap
)

OUT = ensure_dir("results_extras")

def simulate(P, t_end=180.0, dt=0.05):
    y0 = initial_state(P)
    T, Y = integrate(rhs, 0.0, t_end, dt, y0, P)
    y, f, s = series_dict(T, Y, P, pr)
    return T, y, f, s

def panels_extras(tag, T, y, f, s):
    d = ensure_dir(f"{OUT}/{tag}")
    # 1) Stacked fluxes
    stacked_fluxes(T, f, d, tag)
    # 2) Phase-planes
    phase_plane(T, y["y1"], f["NHGB"], "Glucose [mg/dl]", "NHGB [mg/min]", f"{d}/{tag}_phase_y1_vs_NHGB.png")
    phase_plane(T, y["y4"], f["F4"],   "Interstitial insulin [µU/ml]", "Peripheral ID (F4) [mg/min]", f"{d}/{tag}_phase_y4_vs_F4.png")
    phase_plane(T, y["y3"], f["F1"],   "Portal insulin [µU/ml]", "Hepatic production (F1) [mg/min]", f"{d}/{tag}_phase_y3_vs_F1.png")
    # 3) Secretion & pools
    secretion_panels(T, s, y, d, tag)
    # 4) Compartment ratio
    ratios_and_times(T, y, d, tag)

def dose_response(tag, build_P, doses=(0.2,0.25,0.33,0.4,0.5)):
    """
    Re-run with different IVGTT doses (g/kg).
    Implemented by temporarily monkey-patching pr.inputs via a local wrapper.
    """
    peaks_g = []; peaks_i = []

    # Keep original Z1, then wrap
    from inputs import Z1_glucose_input as Z1_orig
    def Z1_scaled(t, BW_kg, gkg):  # local helper
        total_mg = gkg * BW_kg * 1000.0
        return total_mg/3.0 if (0.0 <= t < 3.0) else 0.0

    # Swap in model.rhs by closure? Easier: temporarily replace inputs.Z1 in runtime.
    import inputs as inputs_mod
    for gkg in doses:
        # patch
        inputs_mod.Z1_glucose_input = lambda t, BW_kg, gkg=gkg: Z1_scaled(t, BW_kg, gkg)
        # run
        P = build_P()
        T, y, f, s = simulate(P)
        peaks_g.append(float(np.max(y["y1"])))
        peaks_i.append(float(np.max(y["y2"])))

    # restore
    inputs_mod.Z1_glucose_input = Z1_orig

    dose_response_plot(np.array(doses), np.array(peaks_g), np.array(peaks_i),
                       ensure_dir(f"{OUT}/{tag}"), f"{tag}")

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
            Z[j,i] = y["y1"][np.argmin(np.abs(T - t_eval))]

    sensitivity_heatmap(X, Y, Z, xlab=xsetter.__name__, ylab=ysetter.__name__,
                        title=f"Sensitivity: G({t_eval} min)", outpath=f"{OUT}/{tag}_sens_heatmap.png")

if __name__ == "__main__":
    # ---- CONTROL ----
    P_ctrl = build_control()
    T, y, f, s = simulate(P_ctrl)
    panels_extras("control", T, y, f, s)

    # Dose–response (control)
    dose_response("control", build_control)

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

    # Dose–response (t2d)
    dose_response("t2d", build_t2d)

    # Sensitivity (t2d): hepatic production level (a11) × glucagon inhibition shift (c71)
    xs = np.linspace(5.0, 7.5, 13)   # a11
    ys = np.linspace(110.0, 130.0, 13)  # c71
    def set_a11(P, v): setattr(P, "a11", float(v))
    def set_c71(P, v): setattr(P, "c71", float(v))
    set_a11.__name__ = "a11 (hepatic production scale)"
    set_c71.__name__ = "c71 (insulin inhibition shift)"
    sensitivity_grid("t2d", build_t2d, xs, ys, set_a11, set_c71)

    print("Extra analyses saved to ./results_extras")
