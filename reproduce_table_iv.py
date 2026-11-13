"""Recompute the three rows of Table IV from the Case 2 trajectories."""
import csv

import numpy as np

from make_comparison_figures import DATA_DIR, load, sharing_error_trace
from run_comparison import metrics


def table_values(data):
    # The two values at each step have the same time coordinate. The
    # trapezoidal rule therefore integrates each continuous segment separately.
    time, sharing = sharing_error_trace(data)
    delays = metrics(data)["joint_settling_delays_s"]
    settling_ms = None if any(delay is None for delay in delays) else 1000 * max(delays)
    return (
        float(np.trapezoid(sharing, time)),
        float(np.trapezoid(np.abs(data["voltage_error"]), data["t"])),
        settling_ms,
    )


def main():
    proposed = load(2, "")
    baseline = load(2, "_tcst")
    ablation = load(2, "_no_observer")
    for comparison in (baseline, ablation):
        np.testing.assert_allclose(comparison["t"], proposed["t"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(comparison["load"], proposed["load"])

    labels = ("Current-sharing error integral (A s)",
              "Weighted voltage error integral (V s)", "Max settling delay (ms)")
    target = DATA_DIR / "table_iv.csv"
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("Metric", "Proposed method", "3SM", "Proposed method without observer", "Reduction (%)"))
        for label, result, reference, fixed in zip(
                labels, table_values(proposed), table_values(baseline), table_values(ablation)):
            if reference is None or result is None:
                raise ValueError("3SM or the proposed controller did not settle in the evaluation window.")
            reduction = 100 * (1 - result / reference)
            writer.writerow((label, result, reference, "Not settled" if fixed is None else fixed, reduction))
            fixed_display = "Not settled" if fixed is None else f"{fixed:.6g}"
            print(f"{label:<43} Proposed method={result:.6g}  3SM={reference:.6g}  "
                  f"Proposed method without observer={fixed_display}  Reduction={reduction:.1f}%")
    print(f"Saved {target.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
