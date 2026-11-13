"""Audit published Fig. 4-5 before authorizing manuscript comparisons.

The 20-ms load transition is inferred independently from the plotted plant
states using integral KCL, not fitted to our controller output. The published
alpha=2400 is unchanged. Unreported Lambda and alpha_r remain assumptions.
PyMuPDF extracts vector paths; no raster digitization or curve smoothing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import least_squares

import run_comparison as comparison

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"data"/"reference_figure_curves"
FIGURES = ROOT/"figures"/"reference_reproduction"
COLORS = ("#0085bc", "#75a71e", "#d35421", "#c78f00")


def extract_source_curves(source_pdf):
    """Axis grids calibrate every coordinate; indices refer to source page 10."""
    with fitz.open(source_pdf) as document:
        drawings = document[9].get_drawings()
    x0, x1 = (drawings[i]["rect"].x0 for i in (71, 73))  # ticks 0 and 1 s
    groups = {
        "V": (range(99, 103), 77, 78, 380, 380.5),
        "I_t": (range(149, 153), 120, 121, 40, 50),
        "I_line": (range(202, 206), 177, 178, 0, 10),
        "u": (range(250, 254), 221, 222, 388, 390),
    }
    curves = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (indices, grid0, grid1, value0, value1) in groups.items():
        y0, y1 = (drawings[g]["rect"].y0 for g in (grid0, grid1))
        curves[key] = []
        for i, index in enumerate(indices):
            drawing = drawings[index]
            assert len(drawing["items"]) > 4000, "Source vector layout changed."
            assert all(item[0] == "l" for item in drawing["items"])
            points = [drawing["items"][0][1]]+[a[2] for a in drawing["items"]]
            times = np.array([(p.x-x0)/(x1-x0) for p in points])
            values = np.array([value0+(p.y-y0)/(y1-y0)*(value1-value0) for p in points])
            assert np.all(np.diff(times) >= 0)
            times, unique = np.unique(times, return_index=True)
            values = values[unique]
            curves[key].append((times, values))
            np.savetxt(OUT/f"published_{key}_{i+1}.csv", np.c_[times, values],
                       delimiter=",", header="time_s,value", comments="")
    return curves


def value(curve, times):
    return np.interp(times, *curve)


def integral(curve, left, right):
    times, values = curve
    inside = (times > left) & (times < right)
    t = np.r_[left, times[inside], right]
    y = np.r_[value(curve, left), values[inside], value(curve, right)]
    return np.trapezoid(y, t)


def infer_load(curves):
    # Finite-volume KCL avoids differentiating the tiny plotted voltage ripple.
    edges = np.arange(0.99, 1.030001, 0.001)
    centers = (edges[1:]+edges[:-1])/2
    loads = []
    for left, right in zip(edges[:-1], edges[1:]):
        currents = np.array([integral(s, left, right) for s in curves["I_t"]])
        lines = np.array([integral(s, left, right) for s in curves["I_line"]])
        change = np.array([value(s, right)-value(s, left) for s in curves["V"]])
        loads.append((currents+comparison.plant.B@lines-comparison.plant.C_T@change)/(right-left))
    loads = np.array(loads)
    initial, delta = comparison.INITIAL_LOAD, comparison.FINAL_LOAD-comparison.INITIAL_LOAD

    def ramp_average(start, duration):
        # Integral of clip((t-start)/duration, 0, 1).
        def primitive(t):
            x = np.maximum(t-start, 0)
            return np.where(x < duration, x*x/(2*duration), x-duration/2)
        return np.diff(primitive(edges))/np.diff(edges)

    def residual(parameters):
        expected = initial+ramp_average(*parameters)[:, None]*delta
        return (expected-loads).ravel()
    fit = least_squares(residual, [1.0, .02], bounds=([.995, .005], [1.005, .04]),
                        xtol=1e-13, ftol=1e-13, gtol=1e-13)
    predicted = initial+ramp_average(*fit.x)[:, None]*delta
    step = initial+(centers >= 1)[:, None]*delta
    np.savetxt(OUT/"loads_from_integral_KCL.csv", np.c_[centers, loads], delimiter=",",
               header="window_center_s,IL1_A,IL2_A,IL3_A,IL4_A", comments="")
    return centers, loads, predicted, {
        "method": "integral capacitor KCL from published voltage, DGU current and line-current vector paths",
        "fitted_start_s": float(fit.x[0]), "fitted_duration_s": float(fit.x[1]),
        "ramp_fit_rmse_A": np.sqrt(np.mean((predicted-loads)**2, axis=0)).tolist(),
        "ideal_step_rmse_A": np.sqrt(np.mean((step-loads)**2, axis=0)).tolist(),
        "status": "inferred from published curves; not an explicitly reported source parameter",
    }


def physical_step_bound():
    """Best possible weighted voltage under any |du_i/dt|<=2400 after a step."""
    m = comparison.plant
    a = np.zeros((20, 20)); rate = np.zeros((20, 4)); load = np.zeros((20, 4))
    ci, li, lli = np.linalg.inv(m.C_T), np.linalg.inv(m.L_T), np.linalg.inv(m.L_LINE)
    a[:4, 4:8] = ci; a[:4, 8:12] = ci@m.B
    a[4:8, :4] = -li; a[4:8, 4:8] = -li@m.R_T; a[4:8, 16:20] = li
    a[8:12, :4] = -lli@m.B.T; a[8:12, 8:12] = -lli@m.R_LINE
    a[12:16, 4:8] = -m.L_GAMMA@m.W
    rate[16:20] = np.eye(4); load[:4] = -ci
    output = np.zeros(20); output[:4] = (1/comparison.W_DIAG)/m.H_W
    delay = .001956
    natural = quad(lambda s: float(output@expm(a*s)@load@(comparison.FINAL_LOAD-comparison.INITIAL_LOAD)),
                   0, delay, epsabs=1e-10)[0]
    capacity = sum(quad(lambda s: abs(float(output@expm(a*s)@rate[:, i])),
                        0, delay, epsabs=1e-12)[0] for i in range(4))*2400
    return {"delay_after_ideal_step_s": delay,
            "weighted_voltage_error_upper_bound_V": natural+capacity,
            "meaning": "any causal controller with the literal voltage-input rate bound must fall below this value"}


def waveform_errors(data, curves):
    mask = (data["t"] >= .99) & (data["t"] <= 1.05)
    errors = {}
    for key in ("V", "I_t", "I_line", "u"):
        reference = np.array([value(s, data["t"]) for s in curves[key]]).T
        error = data[key][mask]-reference[mask]
        errors[key] = {"max_abs_per_channel": abs(error).max(axis=0).tolist(),
                       "rmse_per_channel": np.sqrt(np.mean(error**2, axis=0)).tolist()}
    return errors


def figures(data, curves, centers, loads, predicted, step_data):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 10})
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    keys = ("V", "I_t", "I_line", "u")
    labels = ("PCC voltage (V)", "DGU current (A)", "Line current (A)", "Input voltage (V)")
    for row, (key, ylabel) in enumerate(zip(keys, labels)):
        for col in range(2):
            ax = axes[row, col]
            for i, color in enumerate(COLORS):
                tx, y = curves[key][i]
                ax.plot(tx, y, color=color, ls="--", lw=1.1, alpha=.8)
                ax.plot(data["t"], data[key][:, i], color=color, lw=1.0)
            ax.set_xlim((0, 2.4) if col == 0 else (.997, 1.032))
            ax.set_ylabel(ylabel)
            ax.tick_params(direction="in", top=True, right=True)
            ax.grid(alpha=.15)
            if row == 3:
                ax.set_xlabel("Time (s)")
    axes[0, 0].set_title("Published Scenario 1: full interval")
    axes[0, 1].set_title("Load-transition detail")
    fig.legend(handles=[Line2D([], [], color="0.2", lw=1.2, label="Reproduction, alpha=2400"),
                        Line2D([], [], color="0.2", ls="--", lw=1.2, label="Published PDF vector curves")],
               loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(FIGURES/"Scenario1_published_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for i, color in enumerate(COLORS):
        axes[0].plot(centers, loads[:, i], "o", ms=3.5, color=color)
        axes[0].plot(centers, predicted[:, i], color=color, label=f"Load {i+1}")
    axes[0].set(xlabel="Time (s)", ylabel="Load reconstructed by integral KCL (A)", xlim=(.996, 1.026))
    axes[0].legend(ncol=2, fontsize=9)
    for d, label, color in ((step_data, "Ideal instantaneous step", "#c44b31"),
                            (data, "20-ms transition inferred from PDF", "#0077b6")):
        axes[1].plot(d["t"], d["voltage_error"], color=color, label=label)
    axes[1].set(xlabel="Time (s)", ylabel="Weighted-average voltage deviation (V)", xlim=(.997, 1.035))
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True); ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(FIGURES/"Scenario1_load_diagnosis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(source_pdf):
    curves = extract_source_curves(source_pdf)
    centers, loads, predicted, inference = infer_load(curves)
    print(json.dumps(inference, indent=2), flush=True)
    print("Reference Scenario 1: published alpha, inferred 20-ms transition...", flush=True)
    data = comparison.simulate_tcst(0)
    refined = comparison.simulate_tcst(0, step=comparison.DEFAULT_STEP/2)
    step_data = comparison.simulate_tcst(0, reference_transition=0.0)
    errors = waveform_errors(data, curves)
    numerical = {k: float(abs(data[k]-refined[k]).max()) for k in ("V", "I_t", "u")}
    # Engineering agreement criteria are explicit, not a claim of identical source code.
    limits = {"V": .01, "I_t": .25, "I_line": .2, "u": .6}
    waveform_pass = all(max(errors[k]["max_abs_per_channel"]) < limit for k, limit in limits.items())
    numerical_pass = numerical["V"] < .001 and numerical["I_t"] < .01 and numerical["u"] < .05
    load_pass = abs(inference["fitted_duration_s"]-.02) < .0001 and max(inference["ramp_fit_rmse_A"]) < .02
    record = {
        "implementation_version": comparison.IMPLEMENTATION_VERSION,
        "reference_waveform_check_passed": bool(waveform_pass and numerical_pass and load_pass),
        "exact_author_implementation_recovered": False,
        "source_pdf": source_pdf.name, "source_pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "source_url": "https://iris.unipv.it/retrieve/e1f104fd-0d3c-8c6e-e053-1005fe0aa0dd/A%20Robust%20Consensus%20Algorithm%20for%20Current%20Sharing_TCST_per%20IRIS.pdf",
        "load_inference": inference, "waveform_errors_0p99_to_1p05_s": errors,
        "waveform_max_abs_tolerances": limits, "step_halving_max_abs_differences": numerical,
        "ramp_metrics": comparison.metrics(data), "ideal_step_metrics": comparison.metrics(step_data),
        "ideal_step_physical_bound": physical_step_bound(),
        "design": comparison.design_record(),
        "implementation_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                  for name in ("run_comparison.py", "simulate_dc_microgrid.py", "audit_reference_scenario1.py")},
        "limitations": ["Lambda and alpha_r are supplemental assumptions; not recovered author values",
                        "Original high-frequency input ripple is not reproduced point for point",
                        "PDF vector curves have finite coordinate/export precision",
                        "20-ms transition is inferred from figures, not explicitly stated in the paper"],
    }
    comparison.DATA_DIR.mkdir(exist_ok=True)
    (comparison.DATA_DIR/"reference_scenario1_validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    np.savez_compressed(comparison.DATA_DIR/"reference_scenario1_tcst.npz", **data)
    np.savez_compressed(comparison.DATA_DIR/"reference_scenario1_ideal_step_diagnostic.npz", **step_data)
    figures(data, curves, centers, loads, predicted, step_data)
    print(json.dumps({"passed": record["reference_waveform_check_passed"], "waveform": errors, "numerical": numerical}, indent=2), flush=True)
    if not record["reference_waveform_check_passed"]:
        raise RuntimeError("Original Scenario 1 validation failed; manuscript comparisons remain gated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_pdf", type=Path,
                        help="Path to the accepted author manuscript listed in README.md")
    main(parser.parse_args().reference_pdf)
