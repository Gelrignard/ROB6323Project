"""Distributed aerodynamic wind applied each physics update; stock G1 observation/action unchanged.

3.0-specific step override: mirrors ManagerBasedRLEnv.step from release/3.0.0,
with only one new line in the physics loop. Review against upstream when upgrading.
"""
import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .audit import prescribed_uniform_wind
from .wind_math import wind_drag
from .aerodynamics import box_pressure_wrenches, proxy_box_dimensions, tornado_field_world, power_law_wind_profile


class G1WindSACEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        n = cfg.scene.num_envs
        self.wind_world = torch.zeros((n, 3), device=cfg.sim.device)
        self._wind_from = torch.zeros_like(self.wind_world)
        self._wind_to = torch.zeros_like(self.wind_world)
        self._ramp_fraction = torch.ones((n, 1), device=cfg.sim.device)
        self._wind_age = torch.zeros((n,), device=cfg.sim.device)
        self._wind_speed = torch.zeros(n, device=cfg.sim.device)
        self._wind_phase = torch.rand(n, device=cfg.sim.device) * (2 * math.pi)
        self._wind_time = 0.0
        self._gust_world = torch.zeros((n, 3), device=cfg.sim.device)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._robot = self.scene["robot"]
        if "torso_link" not in self._robot.body_names:
            raise RuntimeError("Expected torso_link in official Isaac Lab 3.0 G1 asset")
        self._torso_idx = self._robot.body_names.index("torso_link")
        if self.cfg.wind_model not in ('torso', 'panels'):
            raise ValueError('wind_model must be torso or panels')
        if self.cfg.wind_field not in ('uniform','gust','stochastic_gust','vortex'):
            raise ValueError('wind_field must be uniform, gust, stochastic_gust or vortex')
        self._box_dimensions = proxy_box_dimensions(self._robot.body_names, device=self.device)
        self._forces = torch.zeros((n, self._robot.num_bodies, 3), device=self.device)
        self._torques = torch.zeros_like(self._forces)
        self._resample(torch.arange(n, device=self.device), immediate=True)

    def enable_prescribed_wind(self, speed_mps, heading_deg=0.0,
                               gust_fraction=0.0, gust_period_s=8.0):
        """Replay the same absolute-time wind sequence independently of resets.

        Use uniform field and optional vertical shear in paired evaluation.
        """
        self._prescribed_wind=dict(speed_mps=float(speed_mps),
            heading_deg=float(heading_deg), gust_fraction=float(gust_fraction),
            gust_period_s=float(gust_period_s))
        self._gust_world.zero_()
        self.wind_world[:] = prescribed_uniform_wind(
            self._wind_time, self.num_envs, device=self.device, **self._prescribed_wind)
        self._wind_speed[:] = float(speed_mps)

    def _resample(self, ids, immediate=False):
        if getattr(self, '_prescribed_wind', None) is not None:
            return
        if ids.numel() == 0:
            return
        speed = torch.empty(ids.numel(), device=self.device).uniform_(self.cfg.wind_min_mps, self.cfg.wind_max_mps)
        speed *= (torch.rand(ids.numel(), device=self.device) >= self.cfg.wind_calm_probability)
        self._wind_speed[ids] = speed
        heading = (torch.rand(ids.numel(), device=self.device) * (2.0 * math.pi)
                   if self.cfg.wind_fixed_heading_deg < 0
                   else torch.full_like(speed, math.radians(self.cfg.wind_fixed_heading_deg)))
        new = torch.stack((speed * torch.cos(heading), speed * torch.sin(heading), torch.zeros_like(speed)), -1)
        self._wind_from[ids] = new if immediate else self.wind_world[ids]
        self._wind_to[ids] = new
        self._ramp_fraction[ids] = 1.0 if immediate else 0.0
        self._wind_age[ids] = 0.0
        self._gust_world[ids] = 0.0
        if immediate:
            self.wind_world[ids] = new

    def _advance_wind(self):
        if getattr(self, '_prescribed_wind', None) is not None:
            self.wind_world[:] = prescribed_uniform_wind(
                self._wind_time + self.step_dt, self.num_envs,
                device=self.device, **self._prescribed_wind)
            return
        dt = self.step_dt
        self._wind_age += dt
        to_change = (self._wind_age >= self.cfg.wind_resample_s).nonzero(as_tuple=True)[0]
        self._resample(to_change, immediate=False)
        self._ramp_fraction.add_(dt / max(self.cfg.wind_ramp_s, 1.0e-6)).clamp_(max=1.0)
        f = self._ramp_fraction
        self.wind_world[:] = (1.0 - f) * self._wind_from + f * self._wind_to
        if self.cfg.wind_field == 'stochastic_gust':
            # Discrete-time exact OU update: stationary variance sigma^2 and
            # exponential correlation e^(-dt/tau); synthetic, not a measured storm.
            tau = max(1.0e-3, self.cfg.wind_gust_tau_s)
            decay = math.exp(-dt/tau)
            variation = math.sqrt(1.0-decay*decay)
            local_mean_speed = torch.linalg.vector_norm(self.wind_world[:, :2], dim=-1, keepdim=True)
            sigma = self.cfg.wind_gust_sigma_fraction * local_mean_speed
            noise = torch.randn_like(self._gust_world[:, :2])
            self._gust_world[:, :2] = decay * self._gust_world[:, :2] + variation * sigma * noise

    def _body_wind(self, body_pos_w):
        """World-frame wind at each link, with optional open-terrain power-law shear.

        Classification targets refer to 10 m height; this approximate wind
        profile lowers the effective wind at G1's body height. It is not CFD.
        """
        if self.cfg.wind_field == 'vortex':
            local = body_pos_w - self.scene.env_origins[:, None, :]
            result = tornado_field_world(local, self._wind_speed[:, None, None],
                                         core_radius=self.cfg.wind_vortex_core_radius_m,
                                         inflow_fraction=self.cfg.wind_vortex_inflow_fraction)
        else:
            result = self.wind_world[:, None, :].expand_as(body_pos_w)
            if self.cfg.wind_field == 'stochastic_gust':
                result = result + self._gust_world[:, None, :]
            elif self.cfg.wind_field == 'gust':
                phase = (2 * math.pi * self._wind_time / max(self.cfg.wind_gust_period_s, 0.01)
                         + self._wind_phase)
                gust = self.cfg.wind_gust_amplitude_mps * torch.sin(phase)
                delta = torch.stack((gust, torch.zeros_like(gust), torch.zeros_like(gust)), -1)
                result = result + delta[:, None, :]
        if self.cfg.wind_shear_exponent > 0:
            height = body_pos_w[..., 2] - self.scene.env_origins[:, None, 2]
            result = power_law_wind_profile(result, height,
                                            ref_height_m=self.cfg.wind_reference_height_m,
                                            exponent=self.cfg.wind_shear_exponent,
                                            min_height_m=self.cfg.wind_shear_min_height_m)
        return result

    def _apply_wind(self):
        if not self.cfg.wind_enable:
            return
        self._forces.zero_()
        self._torques.zero_()
        if self.cfg.wind_model == 'panels':
            # Isaac Lab 3.0 quaternion tensors are xyzw. Surface boxes are
            # proxies with estimated dimensions, NOT exact triangle meshes.
            body_pos = self._robot.data.body_com_pos_w.torch
            wind = self._body_wind(body_pos)
            forces, torques = box_pressure_wrenches(
                wind, body_pos, self._robot.data.body_link_quat_w.torch,
                self._robot.data.body_com_lin_vel_w.torch,
                self._robot.data.body_com_ang_vel_w.torch,
                self._box_dimensions, rho=self.cfg.wind_rho,
                cp=self.cfg.wind_pressure_coefficient)
            self._forces[:] = forces
            self._torques[:] = torques
        else:
            v = self._robot.data.body_com_lin_vel_w.torch[:, self._torso_idx, :]
            wind = self._body_wind(self._robot.data.body_com_pos_w.torch)[:, self._torso_idx, :]
            self._forces[:, self._torso_idx, :] = wind_drag(wind, v, self.cfg.wind_rho,
                                                            self.cfg.wind_cd_area)
        # 0 disables capping. Positive cap is a diagnostic guard, not aerodynamics.
        if self.cfg.wind_max_force_n > 0:
            norms = torch.linalg.vector_norm(self._forces, dim=-1, keepdim=True).clamp_min(1e-7)
            scale = (self.cfg.wind_max_force_n / norms).clamp(max=1)
            self._forces.mul_(scale)
            self._torques.mul_(scale)
        # SET rather than ADD: wrench buffers persist, and ADD would accumulate wind every step.
        self._robot.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=self._forces.contiguous(), torques=self._torques.contiguous(), is_global=True
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids):
            self._resample(env_ids, immediate=True)

    def step(self, action):
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.is_rendering
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        for _ in range(self.cfg.decimation // steps_per_call):
            self._sim_step_counter += steps_per_call
            self.action_manager.apply_action()
            self._apply_wind()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt * steps_per_call)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        # Optional pre-reset telemetry for frozen-policy evaluations.  It is
        # captured AFTER physics/reward/termination, but BEFORE _reset_idx()
        # overwrites fallen robots' state.  Normal SAC training is unaffected.
        if getattr(self, "capture_eval_telemetry", False):
            with torch.no_grad():
                body_pos = self._robot.data.body_com_pos_w.torch
                torso_wind = self._body_wind(body_pos)[:, self._torso_idx, :]
                # Isaac Lab 3.0 data fields can expose Torch via `.torch`;
                # allow plain torch.Tensors as well for compatibility.
                def tensor(x):
                    return x.torch if hasattr(x, "torch") else x
                g_b = tensor(self._robot.data.projected_gravity_b)
                lean_deg = torch.rad2deg(torch.acos((-g_b[:, 2]).clamp(-1.0, 1.0)))
                self.extras["wind_eval_telemetry"] = {
                    "root_pos_local_m": (tensor(self._robot.data.root_pos_w) - tensor(self.scene.env_origins)).detach().clone(),
                    "root_lin_vel_b_mps": tensor(self._robot.data.root_lin_vel_b).detach().clone(),
                    "root_ang_vel_b_rps": tensor(self._robot.data.root_ang_vel_b).detach().clone(),
                    "lean_deg": lean_deg.detach().clone(),
                    "wind_reference_world_mps": self.wind_world.detach().clone(),
                    "wind_target_mps": self._wind_speed.detach().clone(),
                    "wind_torso_world_mps": torso_wind.detach().clone(),
                    "wind_net_force_world_n": self._forces.sum(dim=1).detach().clone(),
                    "wind_sum_link_force_magnitudes_n": torch.linalg.vector_norm(self._forces, dim=-1).sum(dim=1).detach().clone(),
                }
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            if self.cfg.compute_final_obs:
                self.extras["final_obs"] = self.observation_manager.compute()
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        if self.sim.consume_reset_request():
            not_yet_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            not_yet_reset[reset_env_ids.long()] = False
            manual = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
            if len(manual) > 0:
                self.reset_terminated[manual.long()] = True
                self.recorder_manager.record_pre_reset(manual)
                self._reset_idx(manual)
                self.recorder_manager.record_post_reset(manual)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.active_terms:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        for recorder in self.video_recorders:
            recorder.step()
        self._advance_wind()
        self._wind_time += self.step_dt
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
