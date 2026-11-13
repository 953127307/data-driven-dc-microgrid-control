"""Regression checks for the published-amplitude, ideal-sign 3SM simulation."""
import unittest

import numpy as np

import run_comparison as comparison
import simulate_dc_microgrid as plant


class ComparisonRegressionTests(unittest.TestCase):
    def test_published_amplitude_is_fixed(self):
        np.testing.assert_array_equal(comparison.ALPHA, np.full(4, 2400.0))
        np.testing.assert_array_equal(plant.L_GAMMA[0], [10, -10, 0, 0])

    def test_levant_gain_powers(self):
        gain = comparison.LAMBDA
        self.assertAlmostEqual(comparison.LAMBDA_0, 3*gain**(1/3))
        self.assertAlmostEqual(comparison.LAMBDA_1, 1.5*np.sqrt(gain))
        self.assertAlmostEqual(comparison.LAMBDA_2, 1.1*gain)

    def test_implicit_differentiator_preserves_a_linear_signal(self):
        step = 1e-3
        value = np.array([0.1, -0.3, 0.2, -0.4])
        velocity = np.array([1.0, -0.5, 2.0, -1.0])
        estimates = np.r_[value, velocity, np.zeros(4)]
        measured = value+step*velocity
        updated = comparison.observer_update(estimates, measured, step)
        np.testing.assert_allclose(updated[:4], measured, atol=1e-14)
        np.testing.assert_allclose(updated[4:8], velocity, atol=1e-10)
        np.testing.assert_allclose(updated[8:], 0, atol=1e-8)

    def test_implicit_differentiator_satisfies_the_original_inclusion(self):
        step = 1e-3
        estimates = np.zeros(12)
        measured = np.array([0.2, -0.1, 1e-8, -1e-8])
        updated = comparison.observer_update(estimates, measured, step)
        error = updated[:4]-measured
        first = updated[4:8]-comparison.LAMBDA_0*abs(error)**(2/3)*np.sign(error)
        second = (updated[8:]-comparison.CANONICAL_SECOND_GAIN
                  * abs(error)**(1/3)*np.sign(error))
        np.testing.assert_allclose((updated[:4]-estimates[:4])/step, first, atol=1e-7)
        np.testing.assert_allclose((updated[4:8]-estimates[4:8])/step, second, atol=1e-7)
        third = (updated[8:]-estimates[8:])/step
        outside = abs(error)>1e-12
        np.testing.assert_allclose(third[outside],
                                   -comparison.LAMBDA_2*np.sign(error[outside]), atol=1e-8)
        self.assertTrue(np.all(abs(third[~outside])<=comparison.LAMBDA_2+1e-8))

    def test_reference_equilibria_satisfy_the_original_plant(self):
        for load in (plant.D_INITIAL, plant.D_FINAL):
            state = comparison.equilibrium(load)
            voltage = plant.V_STAR+comparison.W_DIAG*(state[:4]+state[12:16])
            derivative = plant.physical_derivatives(
                voltage, state[4:8], state[8:12], plant.V_STAR+state[16:], load)
            for values in derivative:
                np.testing.assert_allclose(values, 0, atol=2e-7)
            self.assertLess(abs(state[12:16].sum()), 1e-9)

    def test_common_physical_and_consensus_initial_states(self):
        for case in (1, 2, 3):
            state, estimates = comparison.initial_state(case)
            with np.load(comparison.DATA_DIR/f"case_{case}.npz") as proposed:
                voltage = plant.V_STAR+comparison.W_DIAG*(state[:4]+state[12:16])
                np.testing.assert_allclose(voltage, proposed["V"][0])
                np.testing.assert_allclose(state[4:8], proposed["I_t"][0])
                np.testing.assert_allclose(state[8:12], proposed["I_line"][0])
                np.testing.assert_allclose(state[12:16], proposed["theta"][0])
            np.testing.assert_array_equal(estimates[:4], state[:4])


if __name__ == "__main__":
    unittest.main()
