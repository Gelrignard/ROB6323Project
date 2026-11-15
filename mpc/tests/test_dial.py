"""DIAL's defining annealing/update behavior, independent of robot tuning."""

import unittest

import numpy as np

from mpc.dial import DialConfig, DIALOptimizer, cost_weights


class DialTests(unittest.TestCase):
    def test_two_annealing_axes_match_upstream_formula(self):
        opt = DIALOptimizer(2, DialConfig(horizon_steps=12, knot_intervals=3,
                                        sigma=.8, horizon_decay=.9, trajectory_decay=.5))
        expected = .8 * .5**np.arange(4)[:, None] * .9**np.arange(3, -1, -1)[None, :]
        schedule = opt.noise_schedule(4)
        np.testing.assert_allclose(schedule, expected)
        self.assertTrue(np.all(np.diff(schedule, axis=0) < 0))
        self.assertTrue(np.all(np.diff(schedule, axis=1) > 0))

    def test_weights_match_official_standardized_reward_softmax(self):
        costs = np.array([2., 5., 1.5, 4., 3.])
        rewards = -costs
        logits = (rewards - rewards[-1]) / rewards.std() / .4
        expected = np.exp(logits - logits.max())
        expected /= expected.sum()
        np.testing.assert_allclose(cost_weights(costs, .4), expected)
        # The objective's units or additive baseline do not change the update.
        np.testing.assert_allclose(cost_weights(7 * costs + 16, .4), expected)

    def test_quadratic_interpolation_and_receding_horizon(self):
        opt = DIALOptimizer(1, DialConfig(horizon_steps=12, knot_intervals=3))
        nodes = (.001 * opt.knot_times**2)[:, None]
        controls = opt.interpolate(nodes)
        np.testing.assert_allclose(controls[:, 0], .001 * opt.control_times**2, atol=1e-14)
        shifted = opt.shift(nodes)
        self.assertAlmostEqual(float(shifted[0, 0]), float(controls[1, 0]))
        self.assertAlmostEqual(float(shifted[-1, 0]), 0.)
        self.assertAlmostEqual(float(opt.shift(nodes, tail_action=np.array([.2]))[-1, 0]), .2)

    def test_freezes_first_knot_and_includes_incumbent(self):
        cfg = DialConfig(samples=10, horizon_steps=6, knot_intervals=3, iterations=1)
        opt = DIALOptimizer(2, cfg, seed=11)
        initial = np.full((4, 2), .1)
        captured = []
        def cost(controls):
            captured.append(controls.copy())
            return np.sum(controls**2, axis=(1, 2))
        result = opt.optimize(initial, cost)
        np.testing.assert_allclose(captured[0][:, 0], np.tile(initial[0], (11, 1)))
        np.testing.assert_allclose(captured[0][-1], opt.interpolate(initial))
        np.testing.assert_allclose(result.nodes[0], initial[0])
        self.assertTrue(np.all(np.abs(result.nodes) <= 1))

    def test_annealed_optimizer_reduces_smooth_objective_and_replays_seed(self):
        cfg = DialConfig(samples=256, horizon_steps=8, knot_intervals=4,
                         iterations=6, temperature=.2, freeze_first=False)
        target = np.full((9, 2), .3)
        cost = lambda u: np.mean((u - target)**2, axis=(1, 2))
        initial = np.zeros((5, 2))
        a = DIALOptimizer(2, cfg, seed=3).optimize(initial, cost)
        b = DIALOptimizer(2, cfg, seed=3).optimize(initial, cost)
        np.testing.assert_array_equal(a.nodes, b.nodes)
        self.assertLess(float(cost(a.controls[None])[0]), .01)
        self.assertEqual(len(a.diagnostics), 6)

    def test_constant_and_invalid_rollouts_never_produce_nan_actions(self):
        cfg = DialConfig(samples=8, horizon_steps=8, knot_intervals=4)
        initial = np.full((5, 2), .1)
        opt = DIALOptimizer(2, cfg, seed=0)
        result = opt.optimize(initial, lambda u: np.ones(len(u)))
        np.testing.assert_array_equal(result.nodes, initial)
        np.testing.assert_array_equal(cost_weights([np.nan, np.inf, 2.], .1), [0., 0., 1.])
        with self.assertRaises(RuntimeError):
            opt.optimize(initial, lambda u: np.full(len(u), np.inf))


if __name__ == "__main__":
    unittest.main()
