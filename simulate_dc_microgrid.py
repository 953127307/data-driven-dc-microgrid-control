"""Reproducible DC-microgrid data-driven DSC simulation.

The reference electrical parameters are used only to simulate the physical plant
and to audit the results. The online controller and observer receive data-derived
weights/kappa, measurements, communication signals, and filter states only.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (0, 3)]
B = np.zeros((N, len(EDGES)))
for k, (tail, head) in enumerate(EDGES):
    B[tail, k] = -1.0
    B[head, k] = 1.0

R_T = np.diag([0.2, 0.3, 0.5, 0.1])
L_T = np.diag([1.8e-3, 2.0e-3, 3.0e-3, 2.2e-3])
C_T = np.diag([2.2e-3, 1.9e-3, 2.5e-3, 1.7e-3])
KAPPA_TRUE = np.diag(np.linalg.inv(C_T))
R_LINE = np.diag([70e-3, 50e-3, 80e-3, 60e-3])
L_LINE = np.diag([2.1e-6, 2.3e-6, 2.0e-6, 1.8e-6])
W = np.diag([1 / 0.4, 1 / 0.2, 1 / 0.15, 1 / 0.25])
V_STAR = 380.0 * np.ones(N)
D_INITIAL = np.array([30.0, 15.0, 30.0, 26.0])
D_FINAL = np.array([40.0, 22.0, 20.0, 31.0])
L_GAMMA = np.array(
    [[10.0, -10.0, 0.0, 0.0],
     [-10.0, 20.0, -10.0, 0.0],
     [0.0, -10.0, 20.0, -10.0],
     [0.0, 0.0, -10.0, 10.0]]
)
A_CONSENSUS = W @ L_GAMMA @ W
H_W = np.ones(N) @ np.linalg.solve(W, np.ones(N))

# Online design choices; none is an electrical parameter or a parameter bound.
# Gains retuned 2026-11-09 against the 3SM baseline (tune_frontier.py,
# data/tuning_frontier.json): lambda1 = -6000 lets the post-step sharing
# recovery stay inside the causal floor |delta lambda_L| (max e_I ties the
# 3SM value at 10 weighted A) and roughly halves the voltage-deviation and
# settling metrics. Case 1 records from the stated initial state, so its
# startup transient appears in the recorded panels; Cases 2-3 record from
# the settled 2-s pre-run, matching the comparison protocol.
LAMBDA1_DESIRED = -6000.0
# lambda2 = -4000 adopted 2026-11-09: the Case-2 step-response evidence is
# numerically unchanged from -8000 (verified sweep), while the Case-1 startup
# u excursion shrinks from 237-744 V to 302-566 V and the startup sharing
# peak from 14 to 10.6 weighted A; the startup voltage-deviation peak rises
# only 0.064 -> 0.086 V, still inside the 0.01 V band within 0.94 ms.
LAMBDA2_DESIRED = -4000.0
TAU_DSC = 5.0e-5
ELL_OBSERVER = 40000.0
BETA_OBSERVER = 240000.0


def physical_derivatives(V, I_t, I_line, u, load):
    V_dot = (I_t + B @ I_line - load) / np.diag(C_T)
    I_t_dot = (-R_T @ I_t - V + u) / np.diag(L_T)
    I_line_dot = (-B.T @ V - R_LINE @ I_line) / np.diag(L_LINE)
    return V_dot, I_t_dot, I_line_dot


def offline_probe_alpha(t):
    return np.array([
        4.0 * np.sin(2 * np.pi * (2.0 + 0.4 * i) * t + 0.3 * i)
        + 2.0 * np.sin(2 * np.pi * (7.0 + 0.3 * i) * t + 0.5 * i)
        for i in range(N)
    ])


def offline_probe_u(t):
    return np.array([
        3.0 * np.sin(2 * np.pi * (3.5 + 0.3 * i) * t + 0.7 * i)
        + 1.5 * np.sin(2 * np.pi * (9.0 + 0.2 * i) * t + 0.2 * i)
        for i in range(N)
    ])


def unpack_offline(y):
    return y[0:4], y[4:8], y[8:12], y[12:16], y[16:20]


def offline_quantities(t, y):
    V, I_t, theta, hat_alpha, I_line = unpack_offline(y)
    I_sigma = -B @ I_line
    theta_dot = -L_GAMMA @ W @ I_t
    x_d = V_STAR + W @ theta
    x_d_dot = -A_CONSENSUS @ I_t
    alpha = (
        D_INITIAL + I_sigma - 0.5 * (V - x_d) + offline_probe_alpha(t)
    )
    hat_alpha_dot = (alpha - hat_alpha) / 2.0e-3
    u = V + 0.1 * (hat_alpha - I_t) + offline_probe_u(t)
    V_dot, I_t_dot, I_line_dot = physical_derivatives(V, I_t, I_line, u, D_INITIAL)
    return {
        "V": V, "I_t": I_t, "theta": theta, "hat_alpha": hat_alpha,
        "I_line": I_line, "I_sigma": I_sigma, "x_d": x_d,
        "x_d_dot": x_d_dot, "hat_alpha_dot": hat_alpha_dot, "u": u,
        "V_dot": V_dot, "I_t_dot": I_t_dot, "I_line_dot": I_line_dot,
    }


def offline_rhs(t, y):
    q = offline_quantities(t, y)
    return np.concatenate((
        q["V_dot"], q["I_t_dot"], -L_GAMMA @ W @ q["I_t"],
        q["hat_alpha_dot"], q["I_line_dot"],
    ))


def synthesize_weights(Z, Psi, Y, lambda_desired):
    m = Psi.shape[0]
    C = np.zeros((m, 1))
    C[0, 0] = 1.0
    p0 = Psi.T @ np.linalg.solve(Psi @ Psi.T, C)
    Psi_ker = np.eye(Psi.shape[1]) - Psi.T @ np.linalg.solve(Psi @ Psi.T, Psi)
    direction = Z @ Psi_ker
    direction_norm = np.linalg.norm(direction)
    Zp0 = (Z @ p0).item()
    lambda_target = min(lambda_desired, Zp0 - 1.0)
    chi = (Zp0 - lambda_target) / direction_norm
    p_star = p0 - chi * direction.T / direction_norm
    G = np.vstack((Z, Psi))
    Q_star = G.T @ np.linalg.solve(
        G @ G.T, np.vstack((np.zeros((2, m - 1)), np.eye(m - 1)))
    )
    T = np.column_stack((p_star, Q_star))
    w_T = Y @ T
    return w_T, {
        "rank_Psi": int(np.linalg.matrix_rank(Psi)),
        "rank_G": int(np.linalg.matrix_rank(G)),
        "condition_G": float(np.linalg.cond(G)),
        "lambda_target": float(lambda_target),
        "lambda_reconstructed": float((Z @ p_star).item()),
        "Psi_T_identity_error": float(np.linalg.norm(Psi @ T - np.eye(m))),
        "Z_T_identity_error": float(
            np.linalg.norm(Z @ T - np.hstack((lambda_target, np.zeros(m - 1))))
        ),
    }


def generate_offline_data_and_weights():
    y0 = np.concatenate((V_STAR, D_INITIAL, np.zeros(N), D_INITIAL, np.zeros(len(EDGES))))
    solution = solve_ivp(
        offline_rhs, (0.0, 1.0), y0, method="LSODA", dense_output=True,
        rtol=2e-10, atol=2e-11, max_step=2e-4,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    # Derivative-free integral identification on a dense offline data grid.
    dense_t = np.linspace(0.0, 1.0, 10001)
    dense_y = solution.sol(dense_t)
    H = np.zeros((N, 20))
    delta_V = np.zeros((N, 20))
    window_indices = np.linspace(0, len(dense_t) - 1, 21, dtype=int)
    kappa_hat = np.zeros(N)
    for i in range(N):
        q_values = []
        for k in range(len(dense_t)):
            q = offline_quantities(dense_t[k], dense_y[:, k])
            q_values.append(q["I_t"][i] - q["I_sigma"][i] - D_INITIAL[i])
        integral = cumulative_trapezoid(q_values, dense_t, initial=0.0)
        for w in range(20):
            a, b = window_indices[w], window_indices[w + 1]
            H[i, w] = integral[b] - integral[a]
            delta_V[i, w] = dense_y[i, b] - dense_y[i, a]
        kappa_hat[i] = np.sum(H[i] * delta_V[i]) / np.sum(H[i] ** 2)

    sample_t = np.linspace(0.02, 0.98, 240)
    sample_y = solution.sol(sample_t)
    quantities = [offline_quantities(t, y) for t, y in zip(sample_t, sample_y.T)]
    weights_1, weights_2, data_checks = [], [], []
    for i in range(N):
        Z1 = np.array([[q["V_dot"][i] - q["x_d_dot"][i] for q in quantities]])
        Psi1 = np.array([
            [q["V"][i] - q["x_d"][i] for q in quantities],
            [q["I_sigma"][i] for q in quantities],
            [D_INITIAL[i] for q in quantities],
            [q["x_d_dot"][i] for q in quantities],
        ])
        Y1 = np.array([[q["I_t"][i] for q in quantities]])
        w1, check1 = synthesize_weights(Z1, Psi1, Y1, LAMBDA1_DESIRED)

        Z2 = np.array([[q["I_t_dot"][i] - q["hat_alpha_dot"][i] for q in quantities]])
        Psi2 = np.array([
            [q["I_t"][i] - q["hat_alpha"][i] for q in quantities],
            [q["V"][i] for q in quantities],
            [q["I_t"][i] for q in quantities],
            [q["hat_alpha_dot"][i] for q in quantities],
        ])
        Y2 = np.array([[q["u"][i] for q in quantities]])
        w2, check2 = synthesize_weights(Z2, Psi2, Y2, LAMBDA2_DESIRED)
        weights_1.append(w1)
        weights_2.append(w2)
        data_checks.append({"DGU": i + 1, "stage_1": check1, "stage_2": check2})

    return np.vstack(weights_1), np.vstack(weights_2), kappa_hat, data_checks


def unpack_closed(y):
    return {
        "V": y[0:4], "I_t": y[4:8], "theta": y[8:12],
        "hat_V": y[12:16], "hat_d": y[16:20],
        "hat_alpha": y[20:24], "I_line": y[24:28],
    }


def closed_quantities(t, y, load, w1, w2, kappa_hat):
    s = unpack_closed(y)
    I_sigma = -B @ s["I_line"]
    theta_dot = -L_GAMMA @ W @ s["I_t"]
    x_d = V_STAR + W @ s["theta"]
    x_d_dot = -A_CONSENSUS @ s["I_t"]
    z1 = s["V"] - x_d
    z2 = s["I_t"] - s["hat_alpha"]
    q = s["I_t"] - I_sigma
    hat_V_dot = kappa_hat * (q - s["hat_d"]) + ELL_OBSERVER * (s["V"] - s["hat_V"])
    hat_d_dot = -BETA_OBSERVER * (s["V"] - s["hat_V"])
    alpha = np.array([
        w1[i] @ np.array([z1[i], I_sigma[i], s["hat_d"][i], x_d_dot[i]])
        for i in range(N)
    ])
    hat_alpha_dot = (alpha - s["hat_alpha"]) / TAU_DSC
    u = np.array([
        w2[i] @ np.array([z2[i], s["V"][i], s["I_t"][i], hat_alpha_dot[i]])
        for i in range(N)
    ])
    V_dot, I_t_dot, I_line_dot = physical_derivatives(s["V"], s["I_t"], s["I_line"], u, load)
    s.update({
        "I_sigma": I_sigma, "theta_dot": theta_dot, "x_d": x_d,
        "x_d_dot": x_d_dot, "z1": z1, "z2": z2, "alpha": alpha,
        "hat_alpha_dot": hat_alpha_dot, "u": u, "V_dot": V_dot,
        "I_t_dot": I_t_dot, "I_line_dot": I_line_dot,
        "hat_V_dot": hat_V_dot, "hat_d_dot": hat_d_dot,
    })
    return s


def closed_rhs(t, y, load, w1, w2, kappa_hat):
    q = closed_quantities(t, y, load, w1, w2, kappa_hat)
    return np.concatenate((
        q["V_dot"], q["I_t_dot"], q["theta_dot"], q["hat_V_dot"],
        q["hat_d_dot"], q["hat_alpha_dot"], q["I_line_dot"],
    ))


def integrate_closed(t_span, y0, load, w1, w2, kappa_hat, t_eval=None):
    result = solve_ivp(
        closed_rhs, t_span, y0, args=(load, w1, w2, kappa_hat),
        method="LSODA", t_eval=t_eval, rtol=2e-9, atol=2e-10, max_step=1e-4,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def initial_closed_state():
    # Only the pre-run starts with theta(-2)=0. The displayed experiment
    # continues from the pre-run terminal state without resetting any state.
    return np.concatenate((
        V_STAR, D_INITIAL, np.zeros(N), V_STAR, D_INITIAL,
        D_INITIAL, np.zeros(len(EDGES)),
    ))


def main():
    import sys
    from run_scenarios import run_study
    run_study(sys.modules[__name__])


if __name__ == "__main__":
    main()
