"""Plot the Case 2 comparison in Figure 10 from the saved traces."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures" / "r0_style"

# lw explicit: black ours 1.0 (user request Nov 9), red 3SM stays 1.5.
OURS = {"color": "#000000", "ls": "-", "lw": 1.0, "zorder": 3}
TCST = {"color": "#CC0000", "ls": "--", "lw": 1.5, "zorder": 2}
ABLATION = {"color": "#7A1FA2", "ls": "-.", "lw": 1.4, "zorder": 1}


def load(case, suffix):
    with np.load(DATA_DIR / f"case_{case}{suffix}.npz") as d:
        if suffix == "_tcst" and (
            "implementation_version" not in d or int(d["implementation_version"]) != 5
        ):
            raise ValueError(
                f"Case {case} 3SM data do not match the published-amplitude ideal-sign implementation. "
                "Regenerate the comparison after validating the corrected controller design."
            )
        return {k: d[k] for k in d.files}


def style_axis(ax):
    # Line widths target the FINAL print scale (width=\columnwidth):
    # measured in Manuscript.pdf, the Fig_A series prints frames at ~0.47 pt
    # (MATLAB 1 pt x 0.47 canvas->column shrink) while this 368.5 pt-wide
    # canvas shrinks x0.68, so 0.70 reproduces the same 0.47 pt on paper
    # (user: right figure's frame thicker than the left ones).
    for spine in ax.spines.values():
        spine.set_linewidth(0.70)
    ax.tick_params(direction="in", top=True, right=True, length=5, width=0.70,
                   labelsize=12)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))



def render_reference_scenario():
    from run_comparison import DATA_DIR, W_DIAG
    with np.load(DATA_DIR / "reference_scenario1_tcst.npz") as record:
        data = {key: record[key] for key in record.files}
    colors = ("#35b878", "#3e2b6d", "#0072BD", "#D95319")
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 8.0), sharex=True)
    voltage_average = data["V"] @ (1/W_DIAG) / np.sum(1/W_DIAG)
    states = (data["V"], data["I_t"], data["I_line"], data["u"])
    labels = ("V (V)", "It (A)", "Line current (A)", "u (V)")
    for panel, (ax, values, ylabel) in enumerate(zip(axes, states, labels)):
        for i in range(4):
            name = str(i+1) if panel != 2 else ("12", "23", "34", "14")[i]
            ax.plot(data["t"], values[:, i], color=colors[i], label=name, lw=1.3)
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlim(0, 2.4)
        ax.axvline(1, color="0.7", lw=0.6)
        ax.legend(ncol=4, fontsize=9, loc="best", frameon=False)
    axes[0].plot(data["t"], voltage_average, "k--", lw=1, label="weighted mean")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Scenario 1: alpha=2400, 20-ms load transition inferred from PDF", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Reference_3SM_Scenario1.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Reference_3SM_Scenario1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def sharing_error_trace(data):
    """Insert the left limit at each load step, before the saved right limit."""
    from simulate_dc_microgrid import H_W

    time = data["t"]
    error = data["sharing_error"]
    load_steps = np.diff(data["load"], axis=0)
    jumps = np.flatnonzero(np.any(load_steps != 0, axis=1)) + 1
    target_steps = load_steps[jumps - 1].sum(axis=1) / H_W
    # The inductor currents are continuous. Undo the change in sharing target
    # at the same time to recover the left limit of each error component.
    left_limit = np.max(np.abs(error[jumps] + target_steps[:, None]), axis=1)
    envelope = np.max(np.abs(error), axis=1)
    return np.insert(time, jumps, time[jumps]), np.insert(envelope, jumps, left_limit)


def render_case2_objectives():
    """Figure 10: Case 2 comparison with the observer-compensation ablation.
    Top panel -- Objective 1 (Proportional Current Sharing): the sharing-error
    envelope max_i|e_Ii| in weighted A, matching the Table IV integral metric.
    Bottom panel -- Objective 2 (Weighted Voltage Balancing): R_V in V.
    Each panel carries an inset zoom of the first load step, mirroring the
    Fig. A r0 style."""
    dsc = load(2, "")
    tcst = load(2, "_tcst")
    no_observer = load(2, "_no_observer")
    ours = (sharing_error_trace(dsc), (dsc["t"], dsc["voltage_error"]))
    theirs = (sharing_error_trace(tcst), (tcst["t"], tcst["voltage_error"]))
    ablation = (sharing_error_trace(no_observer),
                (no_observer["t"], no_observer["voltage_error"]))
    ylabels = (r"$\max_{i}|e_{Ii}|$ (A)", r"$\mathcal{R}_V$ (V)")
    # The purple dashed-dotted curve is the observer-compensation ablation.
    # Its 17.3-A current spike and post-step voltage offset remain inside the
    # existing y limits, so all three schemes use a common axis for comparison.
    ylims = ((-0.7, 35.0), (-2.8, 12.2))
    yticks = (tuple(range(0, 31, 10)), tuple(range(-2, 11, 4)))
    curves = (
        ("Proposed method", OURS, ours),
        ("3SM", TCST, theirs),
        ("Proposed method without observer", ABLATION, ablation),
    )

    # Panel geometry replicates the Fig. A MATLAB canvas (axes width 0.875,
    # height 0.220 on a square canvas -> 4:1 panels as in Fig. 7).
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 3.191))
    for panel, (ax, ylabel, (lo, hi), yv, tag) in enumerate(zip(
            axes, ylabels, ylims, yticks, ("(a)", "(b)"))):
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(False)
        style_axis(ax)
        ax.set_ylim(lo, hi)
        ax.set_yticks(yv)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.text(0.02, 0.92, tag, transform=ax.transAxes,
                ha="left", va="center", fontsize=12,
                fontfamily="Times New Roman")

        left, right = 1.998, 2.035
        inset = ax.inset_axes([0.50, 0.40, 0.46, 0.30])
        for spine in inset.spines.values():
            spine.set_linewidth(0.34)
        inset.set_xlim(left, right)
        inset.tick_params(labelsize=9, direction="in", top=True, right=True,
                          width=0.34)
        inset.ticklabel_format(useOffset=False, axis="x")
        inset.set_facecolor("white")
        inset.set_xticks((2.0, 2.015, 2.030))
        if panel == 0:
            inset.set_ylim(-2, 20)
            inset.set_yticks((0, 10, 20))
        else:
            inset.set_ylim(-2.0, 0.25)
            inset.set_yticks((-2, -1, 0))

        # Retain both one-sided values at each load step in the current plots.
        line_options = {"markevery": []} if panel == 0 else {}
        for label, style, traces in curves:
            time, values = traces[panel]
            line, = ax.plot(time, values, label=label, **style, **line_options)
            mask = (time >= left) & (time <= right)
            zoom_line, = inset.plot(time[mask], values[mask], **style, **line_options)
            if panel == 0:
                line.get_path().should_simplify = False
                zoom_line.get_path().should_simplify = False

    # Keep the three legend entries on one row in each panel.
    for ax in axes:
        lg = ax.legend(fontsize=7.5, frameon=True, edgecolor="black", loc="upper right",
                       ncol=3, borderpad=0.25, columnspacing=0.7, labelspacing=0.28,
                       handlelength=1.5, handleheight=0.8)
        lg.get_frame().set_linewidth(0.67)

    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.105, top=0.955,
                        hspace=0.52)
    name = "Fig_C_1_case2_comparison.pdf"
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    fig.savefig(OUT_DIR / name.replace(".pdf", ".png"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("Exported", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true",
                        help="Also plot the reference waveform generated by the optional audit")
    args = parser.parse_args()
    from run_comparison import require_reference_validation
    require_reference_validation()
    if args.reference and not (DATA_DIR / "reference_scenario1_tcst.npz").exists():
        parser.error("Run audit_reference_scenario1.py with the reference PDF first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_case2_objectives()
    if args.reference:
        render_reference_scenario()


if __name__ == "__main__":
    main()
