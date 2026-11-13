"""Three load cases and a Case 2 run with the load estimate held fixed.

Run simulate_dc_microgrid.py, then make_figures_r0.m in MATLAB. Actual loads are
used only in the plant and in evaluation, never in the online control law.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat

ROOT = Path(__file__).resolve().parent
DURATION = 10.0
EVENTS = np.array([2.0, 4.0, 6.0, 8.0])
AMPLITUDES = np.array([6.0, 4.0, 5.0, 3.0])
FREQUENCIES = np.array([0.5, 0.7, 0.9, 1.1])
PHASES = np.array([0.0, np.pi / 3, 2 * np.pi / 3, np.pi])
VOLTAGE_BAND = 0.01  # V, normalized weighted-average voltage error
SHARING_BAND = 0.05  # weighted A, maximum component of W I_t - lambda_d 1
OBSERVER_BAND = 0.1  # A, maximum absolute load-estimation error


def sample_grid(events):
    parts = [np.linspace(0, DURATION, 5001)]
    for event in [0.0, *events]:
        parts += [event + np.linspace(0, 0.003, 301),
                  event + np.linspace(0.003, 0.08, 386)]
    t = np.unique(np.round(np.concatenate(parts), 10))
    return t[(t >= 0) & (t <= DURATION)]


def initial_state(p, w1):
    # Manuscript Case 1 initial state at rated voltage (V(0)=V*), sharing-
    # consistent I_t(0), and UNBIASED observer initialization V_hat(0)=V(0),
    # I_L_hat(0)=I_L^0 (ablation probes: observer offsets added 96 V to the
    # u spike and the 5 A observer-error excursion without control benefit).
    # Case 1 records its 10-s window from this state, so the startup demand
    # (u peak ~744 V) is visible in the Case-1 panels; the 2-s pre-run washes
    # the startup out before Cases 2-3 (and the 3SM comparison) start at the
    # reached steady state, preserving the comparison protocol.
    voltage = p.V_STAR
    current = (p.D_INITIAL.sum() / p.H_W) * np.diag(np.linalg.inv(p.W))
    theta = np.zeros(p.N)
    hat_voltage = voltage
    hat_load = p.D_INITIAL
    line_current = np.zeros(len(p.EDGES))
    reference_dot = -p.A_CONSENSUS @ current
    bases = np.column_stack((voltage - p.V_STAR,
                             -p.B @ line_current, hat_load, reference_dot))
    hat_alpha = np.einsum('ij,ij->i', w1, bases)  # r(0)=0, not z_2(0)=0
    return np.concatenate((voltage, current, theta, hat_voltage, hat_load,
                           hat_alpha, line_current))


def step_levels(p):
    levels = [p.D_INITIAL.copy()]
    for i, change in enumerate(p.D_FINAL - p.D_INITIAL):
        value = levels[-1].copy()
        value[i] += change
        levels.append(value)
    return np.asarray(levels)


def integrate(p, span, y0, load_function, weights, refined=False, *, observer_enabled=True):
    w1, w2, kappa = weights

    def rhs(t, y):
        derivative = p.closed_rhs(t, y, load_function(t), w1, w2, kappa)
        if not observer_enabled:
            observer_derivative = p.unpack_closed(derivative)
            observer_derivative['hat_V'].fill(0.0)
            observer_derivative['hat_d'].fill(0.0)
        return derivative

    solution = solve_ivp(
        rhs, span, y0, method='LSODA', dense_output=True,
        rtol=2e-10 if refined else 2e-9,
        atol=2e-11 if refined else 2e-10,
        max_step=2.5e-4 if refined else 1e-3,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def simulate(p, case, y0, weights, refined=False, *, observer_enabled=True):
    if not observer_enabled:
        y0 = y0.copy()
        p.unpack_closed(y0)['hat_d'][:] = p.D_INITIAL
    events = EVENTS if case == 2 else np.array([])
    t = sample_grid(events)
    if case == 2:
        levels = step_levels(p)
        Y = np.empty((len(y0), len(t)))
        boundaries = np.r_[0.0, EVENTS, DURATION]
        state = y0.copy()
        for k, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            load_value = levels[k].copy()
            # Integrate a fixed load up to each event, then retain all states.
            sol = integrate(p, (left, right), state,
                            lambda _, d=load_value: d, weights, refined,
                            observer_enabled=observer_enabled)
            mask = (t >= left) & ((t < right) if k < 4 else (t <= right))
            Y[:, mask] = sol.sol(t[mask])
            state = sol.y[:, -1]
        load = levels[np.searchsorted(EVENTS, t, side='right')]
    else:
        def load_at(time):
            if case == 1:
                return p.D_INITIAL
            envelope = (1.0 - np.exp(-time / 0.2)) ** 2
            return p.D_INITIAL + envelope * AMPLITUDES * np.sin(
                2 * np.pi * FREQUENCIES * time + PHASES)
        sol = integrate(p, (0, DURATION), y0, load_at, weights, refined,
                        observer_enabled=observer_enabled)
        Y = sol.sol(t)
        load = np.asarray([load_at(time) for time in t])
    w1, w2, kappa = weights
    records = [p.closed_quantities(time, state, d, w1, w2, kappa)
               for time, state, d in zip(t, Y.T, load)]
    data = {key: np.asarray([r[key] for r in records])
            for key in ('V', 'I_t', 'I_line', 'theta', 'hat_d', 'u', 'z1', 'z2')}
    lam = load.sum(axis=1) / p.H_W
    data.update(t=t, Y=Y.T, load=load, case=case,
                lambda_d=lam, desired_current=lam[:, None] / np.diag(p.W),
                sharing_error=data['I_t'] @ p.W - lam[:, None],
                voltage_deviation=data['V'] - p.V_STAR,
                voltage_error=(data['V'] - p.V_STAR) @
                              np.linalg.solve(p.W, np.ones(p.N)) / p.H_W,
                observer_error=load - data['hat_d'])
    assert np.isfinite(Y).all() and all(np.isfinite(data[k]).all()
                                     for k in ('u', 'sharing_error', 'voltage_error'))
    return data


def settling_delay(t, violation, start, end):
    mask = (t >= start) & (t < end if end < DURATION else t <= end)
    local_t = t[mask]
    bad = np.flatnonzero(violation[mask])
    if len(bad) == 0:
        return 0.0
    next_index = bad[-1] + 1
    return None if next_index >= len(local_t) else float(local_t[next_index] - start)


def metrics(data):
    t = data['t']
    es = np.max(np.abs(data['sharing_error']), axis=1)
    ev = np.abs(data['voltage_error'])
    eo = np.max(np.abs(data['observer_error']), axis=1)
    late = t >= 8.5
    time_rms = lambda v: float(np.sqrt(np.trapezoid(v[late] ** 2, t[late]) /
                                     (t[late][-1] - t[late][0])))
    # RMS of the vector 2-norm uses integration, not unweighted sample averages:
    # the time grid is intentionally denser around load jumps.
    result = {
        'case': int(data['case']),
        'max_abs_weighted_voltage_error_V': float(ev.max()),
        'max_component_sharing_error_weighted_A': float(es.max()),
        'max_observer_error_A': float(eo.max()),
        'voltage_range_V': [float(data['V'].min()), float(data['V'].max())],
        'input_range_V': [float(data['u'].min()), float(data['u'].max())],
        'late_window_s': [8.5, 10.0],
        'late_voltage_rms_V': time_rms(ev),
        'late_sharing_vector_rms_weighted_A': time_rms(np.linalg.norm(data['sharing_error'], axis=1)),
        'late_observer_vector_rms_A': time_rms(np.linalg.norm(data['observer_error'], axis=1)),
        'theta_sum_max_abs': float(np.abs(data['theta'].sum(axis=1)).max()),
    }
    violation = (ev > VOLTAGE_BAND) | (es > SHARING_BAND) | (eo > OBSERVER_BAND)
    if data['case'] in (1, 2):
        starts = np.array([0.0]) if data['case'] == 1 else EVENTS
        ends = np.r_[starts[1:], DURATION]
        result['joint_settling_delays_s'] = [
            {'event_s': float(a), 'delay_s': settling_delay(t, violation, a, b)}
            for a, b in zip(starts, ends)]
    return result


def run_study(p):
    p.DATA_DIR.mkdir(exist_ok=True)
    print('Generating one shared offline data set...', flush=True)
    w1, w2, kappa, checks = p.generate_offline_data_and_weights()
    assert all(c[f'stage_{j}']['rank_G'] == 5 for c in checks for j in (1, 2))
    weights = (w1, w2, kappa)
    y0 = initial_state(p, w1)
    pre = integrate(p, (-2, 0), y0, lambda _: p.D_INITIAL, weights)
    cases = []
    for number in (1, 2, 3):
        print(f'Integrating Case {number}...', flush=True)
        # Case 1 shows the startup from the stated initial state; Cases 2-3
        # start settled, identical to the comparison protocol.
        start = y0 if number == 1 else pre.y[:, -1]
        cases.append(simulate(p, number, start, weights))
    print('Integrating Case 2 with the load estimate held fixed...', flush=True)
    ablation = simulate(p, 2, pre.y[:, -1], weights, observer_enabled=False)
    # A tighter independent integration verifies numerical resolution of the
    # new cases. It is not used as a proof of a stability theorem.
    validation = []
    tight_pre = integrate(p, (-2, 0), y0, lambda _: p.D_INITIAL, weights, True)
    for number, data in enumerate(cases, 1):
        print(f'Checking integration resolution for Case {number}...', flush=True)
        ref = simulate(p, number, y0 if number == 1 else tight_pre.y[:, -1], weights, True)
        differences = {key: float(np.max(np.abs(data[key] - ref[key])))
                       for key in ('V', 'I_t', 'I_line', 'hat_d', 'u')}
        assert differences['V'] < 1e-3 and differences['I_t'] < 1e-3
        assert differences['hat_d'] < 1e-3 and differences['u'] < 0.1
        validation.append({'case': number, 'max_abs_differences': differences})
    expected_w1 = np.array([[checks[i]['stage_1']['lambda_target'] / p.KAPPA_TRUE[i],
                            1.0, 1.0, 1.0 / p.KAPPA_TRUE[i]] for i in range(p.N)])
    expected_w2 = np.array([[checks[i]['stage_2']['lambda_target'] * p.L_T[i, i],
                            1.0, p.R_T[i, i], p.L_T[i, i]] for i in range(p.N)])
    result = {
        'controller_online_uses_physical_parameters': False,
        'same_weights_and_gains_for_all_cases': True, 'comparison_controllers': [],
        'duration_s': DURATION, 'data_checks': checks,
        'weights_1': w1.tolist(), 'weights_2': w2.tolist(), 'kappa_hat': kappa.tolist(),
        'kappa_relative_error_max': float(np.max(np.abs(kappa / p.KAPPA_TRUE - 1))),
        'weight_1_relative_verification_error': float(np.linalg.norm(w1 - expected_w1) / np.linalg.norm(expected_w1)),
        'weight_2_relative_verification_error': float(np.linalg.norm(w2 - expected_w2) / np.linalg.norm(expected_w2)),
        'design': {'lambda1_desired': p.LAMBDA1_DESIRED, 'lambda2_desired': p.LAMBDA2_DESIRED,
                   'tau_DSC': p.TAU_DSC, 'ell_observer': p.ELL_OBSERVER, 'beta_observer': p.BETA_OBSERVER},
        'cold_start_state_prerun': y0.tolist(),
        'initial_state_recorded_window': pre.y[:, -1].tolist(),
        'case2_event_times_s': EVENTS.tolist(), 'case2_load_levels_A': step_levels(p).tolist(),
        'case3': {'amplitude_A': AMPLITUDES.tolist(), 'frequency_Hz': FREQUENCIES.tolist(),
                  'phase_rad': PHASES.tolist(), 'envelope': '(1-exp(-t/0.2))^2'},
        'settling_bands': {'weighted_voltage_V': VOLTAGE_BAND,
                           'max_component_sharing_weighted_A': SHARING_BAND,
                           'max_component_observer_A': OBSERVER_BAND},
        'cases': [metrics(data) for data in cases], 'integration_validation': validation,
    }
    for number, data in enumerate(cases, 1):
        np.savez_compressed(p.DATA_DIR / f'case_{number}.npz', **data, w1=w1, w2=w2, kappa_hat=kappa)
    np.savez_compressed(p.DATA_DIR / 'case_2_no_observer.npz', **ablation,
                        w1=w1, w2=w2, kappa_hat=kappa)
    # MATLAB struct array preserves the original MATLAB figure rendering engine.
    savemat(p.DATA_DIR / 'three_case_results.mat',
            {f'case{number}': data for number, data in enumerate(cases, 1)}, do_compression=True)
    (p.DATA_DIR / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'cases': result['cases'], 'integration_validation': validation}, indent=2), flush=True)


if __name__ == '__main__':
    import simulate_dc_microgrid as plant
    run_study(plant)
