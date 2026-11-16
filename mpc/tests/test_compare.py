"""Comparison layout and truthful recorded-load captions without rendering."""

import unittest

import numpy as np

from mpc.compare import comparison_layout, default_tile_label, exposure_label, pulse_caption, recorded_pulses


class ComparisonTests(unittest.TestCase):
    def buffers(self):
        forces = np.zeros((3, 2, 3))
        torques = np.zeros_like(forces)
        joints = np.zeros((3, 8))
        forces[1, 1] = [6., 8., 0.]
        torques[1, 1] = [0., 0., 3.]
        joints[1, 7] = -4.
        return {"disturbance_sample_time_s": np.array([0., .1, .2]),
                "disturbance_body_forces_N": forces,
                "disturbance_body_torques_Nm": torques,
                "disturbance_joint_torques_Nm": joints}

    def event(self, kind, target, start=.05):
        return {"kind": kind, "target": target, "start_s": start, "duration_s": .25,
                "amplitude": [999.] if kind == "joint_torque" else [999., 0., 0.]}

    def test_three_and_four_case_layouts_preserve_codec_alignment(self):
        self.assertEqual(comparison_layout(3), (3, 1, (2160, 800)))
        self.assertEqual(comparison_layout(4), (2, 2, (1440, 1520)))
        for count in (3, 4):
            self.assertTrue(all(size % 16 == 0 for size in comparison_layout(count)[2]))
        with self.assertRaises(ValueError):
            comparison_layout(2)

    def test_actual_target_loads_are_displayed_once_with_correct_units(self):
        force = self.event("link_force", "torso")
        events = [force, force, self.event("link_torque", "torso"), self.event("joint_torque", "knee")]
        loads = recorded_pulses(self.buffers(), events, .15, 1., {"torso": 1}, {"knee": 7})
        self.assertEqual(len(loads), 3)
        self.assertEqual([row["magnitude"] for row in loads], [10., 3., 4.])
        self.assertEqual([row["unit"] for row in loads], ["N", "N m", "N m"])
        self.assertTrue(all(row["sample_time_s"] == .1 for row in loads))
        self.assertEqual(pulse_caption(loads[0]), "Force on torso: 10.00 N")
        self.assertEqual(pulse_caption(loads[2]), "Joint torque on knee: 4.00 N m")

    def test_future_and_post_termination_events_are_never_shown_active(self):
        event = self.event("link_force", "torso", start=.15)
        # The event is scheduled by .16s, but the latest recorded sample is
        # still .1s: do not invent a force that has not appeared in the log.
        self.assertEqual(recorded_pulses(self.buffers(), [event], .16, 1., {"torso": 1}, {}), [])
        event["start_s"] = 0.
        for when in (.2, .21, .9):
            self.assertEqual(recorded_pulses(self.buffers(), [event], when, .2, {"torso": 1}, {}), [])
        self.assertEqual(recorded_pulses(self.buffers(), [event], -.1, 1., {"torso": 1}, {}), [])

    def test_missing_load_buffer_does_not_substitute_scheduled_peak(self):
        states = {"disturbance_sample_time_s": np.array([0., .1])}
        event = self.event("link_force", "torso")
        load = recorded_pulses(states, [event], .15, 1., {"torso": 1}, {})[0]
        self.assertIsNone(load["magnitude"])
        self.assertIn("recorded magnitude unavailable", pulse_caption(load))

    def test_random_strength_labels_distinguish_the_two_experiments(self):
        moderate = {"scenario": "random", "disturbances": {"strength": .2}}
        large = {"scenario": "random", "disturbances": {"strength": 1.}}
        self.assertIn("0.20", default_tile_label(moderate))
        self.assertIn("1.00", default_tile_label(large))

    def test_native_and_historical_still_air_runs_have_distinct_exposures(self):
        native = {"scenario": "calm", "drag_model": "none", "disturbances": {
            "config": {"enabled": False}, "events": []}}
        old = {**native, "drag_model": "surface"}
        self.assertEqual(exposure_label(native), "Native MuJoCo; no drag or injected loads")
        self.assertEqual(exposure_label(old), "Still-air surface drag; no injected loads")
        native["disturbances"]["config"]["enabled"] = True
        self.assertEqual(exposure_label(native), "Aerodynamics off; injected loads enabled")


if __name__ == "__main__":
    unittest.main()
