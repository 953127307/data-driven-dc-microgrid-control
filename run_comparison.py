"""3SM with the published amplitude and the ideal sign laws of (31)-(33).

Cucuzzella et al., TCST 2019, specify alpha=2400 and the benchmark parameters.
Numerical values of Lambda and the alpha_r design constants are not reported.
The supplemental values below are implementation assumptions, not paper data.

The physical plant uses exact zero-order-hold propagation. A backward-Euler
inclusion integrates the original Levant differentiator. The reaching law
retains its piecewise signs. No terminal feedback or boundary layer is added.

Run run_scenarios.py to generate the proposed-controller traces and
run_comparison.py 2 to reproduce the manuscript comparison. The packaged
reference validation record is checked before the comparison runs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.linalg import expm

import simulate_dc_microgrid as plant
from run_scenarios import AMPLITUDES, EVENTS, FREQUENCIES, PHASES, sample_grid

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMPLEMENTATION_VERSION = 5
REFERENCE_TRANSITION = 0.02  # Inferred from published Fig. 4-5 via integral KCL.
DURATION = 10.0
TIME_SCALE = 1e-3
DEFAULT_STEP = 2.5e-7
ALPHA = np.full(4, 2.4e3)  # Published value, fixed in all runs.
VOLTAGE_BAND, SHARING_BAND = 0.01, 0.05
W_DIAG = np.diag(plant.W).copy()
F = np.linalg.inv(plant.W) @ np.linalg.inv(plant.C_T)
H = F + plant.L_GAMMA @ plant.W
G_D = np.diag(H @ np.linalg.inv(plant.L_T))

# Supplemental assumptions. The paper gives formulas but not these values.
REACHING_FRACTION = 0.25
ALPHA_R_PHYSICAL = REACHING_FRACTION * ALPHA * G_D
ALPHA_R = ALPHA_R_PHYSICAL * TIME_SCALE**3
LAMBDA_PHYSICAL = 1e12
LAMBDA = LAMBDA_PHYSICAL * TIME_SCALE**3
LAMBDA_0 = 3*LAMBDA**(1/3)
LAMBDA_1 = 1.5*np.sqrt(LAMBDA)
LAMBDA_2 = 1.1*LAMBDA
CANONICAL_SECOND_GAIN = LAMBDA_1*np.sqrt(LAMBDA_0)
INITIAL_LOAD = plant.D_INITIAL.copy()
FINAL_LOAD = plant.D_FINAL.copy()
LOAD_AMPLITUDES = AMPLITUDES.copy()
LOAD_FREQUENCIES = 2*np.pi*FREQUENCIES
LOAD_PHASES = PHASES.copy()


def transition_matrices(step):
    """Exact plant/consensus/input-integrator propagation for held load and v."""
    matrix = np.zeros((20, 20))
    matrix[:4, 4:8] = H
    matrix[:4, 8:12] = F @ plant.B
    inverse_l = np.linalg.inv(plant.L_T)
    inverse_line_l = np.linalg.inv(plant.L_LINE)
    matrix[4:8, :4] = -inverse_l @ plant.W
    matrix[4:8, 4:8] = -inverse_l @ plant.R_T
    matrix[4:8, 12:16] = -inverse_l @ plant.W
    matrix[4:8, 16:20] = inverse_l
    matrix[8:12, :4] = -inverse_line_l @ plant.B.T @ plant.W
    matrix[8:12, 8:12] = -inverse_line_l @ plant.R_LINE
    matrix[8:12, 12:16] = -inverse_line_l @ plant.B.T @ plant.W
    matrix[12:16, 4:8] = -plant.L_GAMMA @ plant.W
    inputs = np.zeros((20, 8))
    inputs[:4, :4] = -F
    inputs[16:20, 4:8] = np.eye(4)
    augmented = np.zeros((28, 28))
    augmented[:20, :20] = matrix
    augmented[:20, 20:] = inputs
    propagated = expm(augmented*step)
    return propagated[:20, :20].copy(), propagated[:20, 20:].copy()


def equilibrium(load):
    current = load.sum()/plant.H_W/W_DIAG
    network = plant.B @ np.linalg.inv(plant.R_LINE) @ plant.B.T
    matrix = np.vstack((network, 1/W_DIAG))
    voltage = np.linalg.lstsq(
        matrix, np.r_[current-load, (1/W_DIAG)@plant.V_STAR], rcond=None)[0]
    lines = -np.linalg.solve(plant.R_LINE, plant.B.T@voltage)
    chi = (voltage-plant.V_STAR)/W_DIAG
    applied_input = plant.R_T@current+voltage
    return np.r_[np.zeros(4), current, lines, chi, applied_input-plant.V_STAR]


def initial_state(case):
    if case == 0:
        state = equilibrium(INITIAL_LOAD)
    else:
        with np.load(DATA_DIR/f"case_{case}.npz") as data:
            voltage = data["V"][0].copy()
            current = data["I_t"][0].copy()
            lines = data["I_line"][0].copy()
            chi = data["theta"][0].copy()
            applied_input = data["u"][0]
        sigma = (voltage-plant.V_STAR)/W_DIAG-chi
        state = np.r_[sigma, current, lines, chi, applied_input-plant.V_STAR]
    estimates = np.r_[state[:4], np.zeros(8)]
    return state, estimates


@njit(cache=True)
def observer_update(estimates, measured_sigma, normalized_step):
    """Implicit Euler for the algebraically equivalent canonical form of (33)."""
    updated = np.empty(12)
    h = normalized_step
    for i in range(4):
        residual = (estimates[i]+h*estimates[4+i]+h*h*estimates[8+i]
                    - measured_sigma[i])
        threshold = h**3*LAMBDA_2
        if abs(residual) <= threshold:
            third = estimates[8+i]-residual/(h*h)
            second = estimates[4+i]+h*third
            first = measured_sigma[i]
        else:
            direction = np.sign(residual)
            target = abs(residual)-threshold
            root = target**(1/3)
            for iteration in range(16):
                value = (root**3+h*LAMBDA_0*root**2
                         + h*h*CANONICAL_SECOND_GAIN*root-target)
                slope = (3*root**2+2*h*LAMBDA_0*root
                         + h*h*CANONICAL_SECOND_GAIN)
                proposal = max(0.0, root-value/slope)
                if abs(proposal-root) <= 1e-12*max(root, 1e-12):
                    root = proposal
                    break
                root = proposal
            third = estimates[8+i]-h*LAMBDA_2*direction
            second = estimates[4+i]+h*(third-CANONICAL_SECOND_GAIN*root*direction)
            first = estimates[i]+h*(second-LAMBDA_0*root**2*direction)
        updated[i] = first
        updated[4+i] = second
        updated[8+i] = third
    return updated


@njit(cache=True)
def control(sigma, estimates, reaching_coefficient):
    """Piecewise ideal sign law (31), with the manifolds given after (32)."""
    rates = np.empty(4)
    for i in range(4):
        velocity, acceleration = estimates[4+i], estimates[8+i]
        coefficient = reaching_coefficient[i]
        first_sign = np.sign(acceleration)
        second_sign = np.sign(
            velocity+acceleration**2*first_sign/(2*coefficient))
        root_argument = max(
            second_sign*velocity+acceleration**2/(2*coefficient), 0.0)
        switching = (
            sigma[i]+acceleration**3/(3*coefficient**2)
            + second_sign*(root_argument**1.5/np.sqrt(coefficient)
                           + velocity*acceleration/coefficient))
        at_origin = sigma[i] == 0 and velocity == 0 and acceleration == 0
        on_first = (sigma[i]-acceleration**3/(6*coefficient**2) == 0
                    and velocity+acceleration*abs(acceleration)/(2*coefficient) == 0)
        if at_origin:
            direction = 0.0
        elif on_first:
            direction = first_sign
        elif switching == 0:
            direction = second_sign
        else:
            direction = np.sign(switching)
        rates[i] = -ALPHA[i]*direction
    return rates


@njit(cache=True)
def propagate(state, estimates, step, matrix, inputs, output_steps, case, reaching,
              reference_transition=0.0):
    output = np.empty((len(output_steps), 32))
    output[0, :16] = state[:16]
    output[0, 16:28] = estimates
    output[0, 28:] = state[16:]
    output_index = 1
    rates = control(state[:4], estimates, reaching)
    for index in range(output_steps[-1]):
        midpoint = (index+0.5)*step
        load = INITIAL_LOAD.copy()
        if case == 0 and midpoint >= 1.0:
            fraction = (min((midpoint-1.0)/reference_transition, 1.0)
                        if reference_transition > 0.0 else 1.0)
            load += fraction*(FINAL_LOAD-INITIAL_LOAD)
        elif case == 2:
            for i in range(4):
                if midpoint >= 2*(i+1):
                    load[i] += FINAL_LOAD[i]-INITIAL_LOAD[i]
        elif case == 3:
            envelope = (1-np.exp(-midpoint/0.2))**2
            load += (envelope*LOAD_AMPLITUDES
                     * np.sin(LOAD_FREQUENCIES*midpoint+LOAD_PHASES))
        updated = np.empty(20)
        for i in range(20):
            value = 0.0
            for j in range(20):
                value += matrix[i, j]*state[j]
            for j in range(4):
                value += inputs[i, j]*load[j]+inputs[i, 4+j]*rates[j]
            updated[i] = value
        state = updated
        estimates = observer_update(estimates, state[:4], step/TIME_SCALE)
        rates = control(state[:4], estimates, reaching)
        if output_index < len(output_steps) and index+1 == output_steps[output_index]:
            output[output_index, :16] = state[:16]
            output[output_index, 16:28] = estimates
            output[output_index, 28:] = state[16:]
            output_index += 1
    return output


def simulate_tcst(case, step=DEFAULT_STEP, reaching=None,
                  reference_transition=REFERENCE_TRANSITION):
    if case == 0:
        times = np.unique(np.round(np.r_[
            np.arange(0, 2.40001, 0.001), np.arange(0.998, 1.100001, 1e-5)], 10))
    else:
        times = sample_grid(EVENTS if case == 2 else np.array([]))
    output_steps = np.unique(np.rint(times/step).astype(np.int64))
    times = output_steps*step
    state, estimates = initial_state(case)
    matrix, inputs = transition_matrices(step)
    reaching = ALPHA_R if reaching is None else reaching
    states = propagate(state, estimates, step, matrix, inputs, output_steps, case, reaching,
                       reference_transition)
    voltage = plant.V_STAR+W_DIAG*(states[:, :4]+states[:, 12:16])
    load = np.tile(INITIAL_LOAD, (len(times), 1))
    if case == 0:
        fraction = (np.clip((times-1.0)/reference_transition, 0.0, 1.0)
                    if reference_transition > 0.0 else (times >= 1.0).astype(float))
        load += fraction[:, None]*(FINAL_LOAD-INITIAL_LOAD)
    elif case == 2:
        for i, event in enumerate(EVENTS):
            load[times >= event, i] += (FINAL_LOAD-INITIAL_LOAD)[i]
    elif case == 3:
        envelope = (1-np.exp(-times/0.2))**2
        load += (envelope[:, None]*AMPLITUDES
                 * np.sin(times[:, None]*2*np.pi*FREQUENCIES+PHASES))
    current = states[:, 4:8]
    target = load.sum(axis=1)/plant.H_W
    return {
        "case": case, "t": times, "integration_state": states,
        "V": voltage, "I_t": current, "I_line": states[:, 8:12],
        "chi": states[:, 12:16], "u": plant.V_STAR+states[:, 28:],
        "load": load, "sharing_error": current@plant.W-target[:, None],
        "voltage_error": (voltage-plant.V_STAR)@np.linalg.solve(plant.W, np.ones(4))/plant.H_W,
        "implementation_version": IMPLEMENTATION_VERSION,
        "integration_step_s": step,
        "reference_load_transition_s": reference_transition if case == 0 else 0.0,
    }


def metrics(data):
    times = data["t"]
    voltage = abs(data["voltage_error"])
    sharing = np.max(abs(data["sharing_error"]), axis=1)
    late = times >= (1.8 if data["case"] == 0 else 8.5)
    def rms(values):
        return float(np.sqrt(np.trapezoid(values[late]**2, times[late])
                             / (times[late][-1]-times[late][0])))
    delays = []
    starts = [1.0] if data["case"] == 0 else ([0.0] if data["case"] == 1 else EVENTS)
    if data["case"] != 3:
        for left, right in zip(starts, [*starts[1:], float(times[-1])]):
            mask = (times >= left) & ((times < right) if right < times[-1] else (times <= right))
            bad = np.flatnonzero((voltage[mask] > VOLTAGE_BAND)
                                 | (sharing[mask] > SHARING_BAND))
            delays.append(0.0 if not len(bad) else
                          (None if bad[-1]+1 == mask.sum()
                           else float(times[mask][bad[-1]+1]-left)))
    return {
        "max_abs_weighted_voltage_error_V": float(voltage.max()),
        "max_component_sharing_error_weighted_A": float(sharing.max()),
        "voltage_range_V": [float(data["V"].min()), float(data["V"].max())],
        "input_range_V": [float(data["u"].min()), float(data["u"].max())],
        "late_voltage_rms_V": rms(voltage),
        "late_sharing_vector_rms_weighted_A": rms(np.linalg.norm(data["sharing_error"], axis=1)),
        "joint_settling_delays_s": delays,
    }


def design_record():
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "published_alpha": ALPHA.tolist(), "published_communication_weights": [10, 10, 10],
        "supplemental_assumption_alpha_r": ALPHA_R_PHYSICAL.tolist(),
        "supplemental_assumption_alpha_r_fraction": REACHING_FRACTION,
        "supplemental_assumption_Lambda_physical": LAMBDA_PHYSICAL,
        "time_scale_s": TIME_SCALE, "primary_step_s": DEFAULT_STEP,
        "refined_step_s": DEFAULT_STEP/2, "boundary_layer": None,
        "added_terminal_feedback": False,
        "reference_load_transition_s_inferred_from_figures": REFERENCE_TRANSITION,
        "original_transient_replication": "see reference_scenario1_validation.json; unreported constants remain assumptions",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def require_reference_validation():
    record = json.loads((DATA_DIR/"reference_scenario1_validation.json").read_text(encoding="utf-8"))
    if not record["reference_waveform_check_passed"]:
        raise ValueError("Original Scenario 1 has not passed waveform validation.")
    for filename, digest in record["implementation_sha256"].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{filename} changed: rerun audit_reference_scenario1.py first.")
    return record


def main(cases=(0,)):
    validation = require_reference_validation() if any(cases) else None
    pending = {}
    summaries = {}
    for case in cases:
        print(f"Case {case}: alpha=2400, ideal sign controller...", flush=True)
        data = simulate_tcst(case)
        print(f"Case {case}: half-step verification...", flush=True)
        reference = simulate_tcst(case, step=DEFAULT_STEP/2)
        differences = {key: float(np.max(abs(data[key]-reference[key])))
                       for key in ("V", "I_t", "u", "voltage_error", "sharing_error")}
        if not all(np.isfinite(data[key]).all() for key in ("V", "I_t", "u")):
            raise ValueError(f"Case {case} has nonfinite trajectories.")
        if differences["V"] > 0.01 or differences["I_t"] > 0.1 or differences["u"] > 0.5:
            raise ValueError(f"Case {case}: step resolution is insufficient: {differences}")
        summary = {"3sm": metrics(data), "step_halving_differences": differences}
        if case:
            with np.load(DATA_DIR/f"case_{case}.npz") as ours:
                summary["proposed"] = metrics(dict(ours))
                for field in ("V", "I_t", "I_line"):
                    np.testing.assert_allclose(data[field][0], ours[field][0], atol=1e-9, rtol=0)
        else:
            target = equilibrium(FINAL_LOAD)
            summary["final_equilibrium_current_error_A"] = float(
                np.max(abs(data["I_t"][-1]-target[4:8])))
            summary["published_figure_voltage_axis_V"] = [379.5, 380.5]
            summary["published_transient_match_established"] = bool(
                validation and validation["reference_waveform_check_passed"])
        pending[case] = data
        summaries[f"case_{case}"] = summary
        print(json.dumps(summary, indent=2), flush=True)
    for case, data in pending.items():
        name = "reference_scenario1_tcst.npz" if case == 0 else f"case_{case}_tcst.npz"
        np.savez_compressed(DATA_DIR/name, **data)
    suffix = "" if cases == (0, 1, 2, 3) else "_case_"+"_".join(map(str, cases))
    (DATA_DIR/f"comparison_tcst_metrics{suffix}.json").write_text(
        json.dumps({"design": design_record(), "cases": summaries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(tuple(map(int, sys.argv[1:])) or (0,))
