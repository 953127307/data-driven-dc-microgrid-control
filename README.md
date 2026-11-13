# Data-driven control of DC microgrids

This repository accompanies the manuscript *A Data-Driven Framework for Load Estimation and Distributed Dynamic Surface Control of Direct Current Microgrids*. It contains the four-DGU simulation, the Case 2 comparison of the proposed method, third-order sliding-mode control (3SM), and the proposed method without the load observer, and the scripts for generating the associated figures and numerical results. See [NUMERICAL_SETTINGS.md](NUMERICAL_SETTINGS.md) for the offline excitation, sampling, solver settings, and the parameters used to implement the 3SM baseline.

## Environment

The Python scripts were checked with Python 3.12.4. In a Python 3.12 environment, install the dependencies with

```bash
python -m pip install -r requirements.txt
```

Run the commands below from this directory. The Figure 10 plotting script uses Times New Roman when available. The optional MATLAB figure workflow was written for MATLAB R2023b and uses the TeX Live `epstopdf` command.

## Figure 10 and Table IV from the included results

```bash
python make_comparison_figures.py
python reproduce_table_iv.py
```

Figure 10 is written to `figures/r0_style/Fig_C_1_case2_comparison.pdf` with a PNG preview. The table is printed and saved as `data/table_iv.csv`.

| Metric | Proposed method | 3SM | Proposed method without observer |
| --- | ---: | ---: | ---: |
| Current-sharing error integral (A·s) | 0.03159 | 0.1423 | 0.03596 |
| Weighted voltage error integral (V·s) | 0.000264 | 0.0194 | 3.006 |
| Maximum settling delay (ms) | 3.6 | 12.4 | Not settled |

All three schemes use the same Case 2 load schedule. In the observer ablation, the DSC uses the fixed initial load estimate throughout Case 2. The sharing-error integral uses the one-sided error values at each load step. Settling requires both the current-sharing error and the weighted-average voltage error to enter and stay within their respective 0.05 A and 0.01 V bands through the end of each post-step interval. The ablation does not meet this criterion in any of the four intervals.

## Run the simulations again

```bash
python run_scenarios.py
python -m unittest test_comparison
python run_comparison.py 2
python make_comparison_figures.py
python reproduce_table_iv.py
```

`run_scenarios.py` generates the offline dataset and weights, then runs the proposed controller for constant loads, asynchronous load steps, and continuously varying loads. It also regenerates the Case 2 observer ablation as `data/case_2_no_observer.npz`. `run_comparison.py 2` regenerates the 3SM trace for the asynchronous load steps and repeats the calculation at half the integration step to check numerical resolution. The integration step is much smaller than the recording interval, so this run may take a few minutes. Regeneration replaces the included trajectory files in `data/`.

The `data/` directory includes the manuscript's three proposed-controller traces, the Case 2 3SM and observer-ablation traces, the MATLAB data file, summary metrics, and a reference validation record. The comparison checks that the reference validation record corresponds to the supplied implementation.

## Figures 3 through 9

Use the included `data/three_case_results.mat` or generate it with `python run_scenarios.py`. Then run

```bash
matlab -batch "make_figures_r0"
python finalize_figures.py
```

The MATLAB workflow saves `Fig_A_*.fig` and EPS files in `figures/r0_style/`. The Python finalizer produces PDF and PNG versions in that same directory.

## Optional 3SM reference check

The baseline follows M. Cucuzzella et al., “A Robust Consensus Algorithm for Current Sharing and Voltage Regulation in DC Microgrids,” *IEEE Transactions on Control Systems Technology*, 2019, [DOI: 10.1109/TCST.2018.2834878](https://doi.org/10.1109/TCST.2018.2834878). To repeat the reference check, download the [accepted author manuscript from the University of Pavia repository](https://iris.unipv.it/retrieve/e1f104fd-0d3c-8c6e-e053-1005fe0aa0dd/A%20Robust%20Consensus%20Algorithm%20for%20Current%20Sharing_TCST_per%20IRIS.pdf) and pass its location to the audit script.

```bash
python audit_reference_scenario1.py "/path/to/reference.pdf"
```

The script reads vector paths from the accepted manuscript, estimates the load transition for its reference scenario, and writes a fresh validation record. The original article and extracted vectors are external inputs to this optional check. The packaged validation record suffices to rerun the manuscript comparison.
