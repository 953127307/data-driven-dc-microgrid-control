# Simulation settings

## Offline data collection

For DGU $i$, set $r=i-1$. The probing signals in [simulate_dc_microgrid.py](simulate_dc_microgrid.py) are

$$
p_{\alpha i}(t)=4\sin(2\pi(2+0.4r)t+0.3r)+2\sin(2\pi(7+0.3r)t+0.5r),
$$

$$
p_{u i}(t)=3\sin(2\pi(3.5+0.3r)t+0.7r)+1.5\sin(2\pi(9+0.2r)t+0.2r).
$$

Time is in seconds, $p_{\alpha i}$ is in amperes, and $p_{u i}$ is in volts. With feedback coefficients in SI units, the offline inputs are

$$
\alpha_i^{\mathrm{off}}=I_{Li}^0+I_{\Sigma i}-0.5(V_i-x_{di})+p_{\alpha i},
\qquad
u_i^{\mathrm{off}}=V_i+0.1(\hat\alpha_i^{\mathrm{off}}-I_{ti})+p_{u i}.
$$

The offline filter time constant is $2\times10^{-3}$ s. The initial conditions are $V(0)=V^*$, $I_t(0)=\hat\alpha^{\mathrm{off}}(0)=I_L^0$, $\theta(0)=0$, and $I_\ell(0)=0$.

The 240 sample times are uniformly spaced on $[0.02,0.98]$ s, including both endpoints. Integral regression uses the 20 windows $[0.05(m-1),0.05m]$ s for $m=1,\ldots,20$, with composite trapezoidal quadrature on a $10^{-4}$ s grid.

## Numerical integration for the proposed controller

| Stage | Solver | Relative tolerance | Absolute tolerance | Maximum step (s) |
| --- | --- | ---: | ---: | ---: |
| Offline data collection | LSODA | $2\times10^{-10}$ | $2\times10^{-11}$ | $2\times10^{-4}$ |
| Online closed-loop simulation | LSODA | $2\times10^{-9}$ | $2\times10^{-10}$ | $10^{-3}$ |

The online integration in [run_scenarios.py](run_scenarios.py) restarts at each load step and preserves all dynamic states.

All three load cases use the common online DSC filter time constant $\tau=5\times10^{-5}$ s (50 µs), set by `TAU_DSC` in [simulate_dc_microgrid.py](simulate_dc_microgrid.py).

For the four-DGU benchmark, we formed the state matrix from the affine right-hand side `closed_rhs`, using the controller weights and observer coefficient estimates recorded in [data/metrics.json](data/metrics.json). We projected the matrix onto $\mathbf{1}^{\mathsf T}\theta=0$ using an orthonormal basis. The maximum real part of the eigenvalues of the resulting 27-state matrix was approximately $-2.22145\times10^3\,\mathrm{s}^{-1}$, so the reduced matrix is Hurwitz.

## Case 2 observer ablation

The full controller and the ablation use the same settled Case 2 initial plant, consensus, and DSC filter states. The ablation freezes the observer states and uses the initial load vector $\hat I_L=I_L^0=[30,15,30,26]^{\mathsf T}$ A in the DSC throughout the run. The load profile, controller weights, filter, recording grid, and online solver settings are shared with the full Case 2 run. The weighted-average voltage error is normalized by $\mathbf 1^{\mathsf T}W^{-1}\mathbf 1$ for all three schemes.

## 3SM comparison

[run_comparison.py](run_comparison.py) uses the published switching amplitude $\alpha_i=2400$ and communication weights $\gamma_{12}=\gamma_{23}=\gamma_{34}=10$. The additional settings for this implementation are $\alpha_{ri}=0.25\alpha_i G_{d,ii}$ and $\Lambda_i=10^{12}$ in physical time. The plant uses exact propagation with held input rates, and the Levant differentiator uses a backward-Euler differential inclusion. The time step is 0.25 µs, with a check at 0.125 µs. The 3SM differentiator starts at the measured sliding variable with zero derivative estimates. [README.md](README.md) describes the reference validation and the commands for running the comparison.

For Case 2, the 3SM input integrator starts at the proposed controller’s input at the common initial state, `u[0]` in [data/case_2.npz](data/case_2.npz). Halving the 3SM integration step from 0.25 µs to 0.125 µs preserved all four joint settling delays, with a maximum of 12.4 ms. The Table 4 current-sharing and weighted voltage error integrals changed by 0.092% and 0.0057%, respectively, relative to the 0.25 µs run.
