"""Preserve the stock 3.0 G1 observation/action interface to run its pretrained actor."""
from isaaclab.utils import configclass
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg


@configclass
class G1WindSACEnvCfg(G1FlatEnvCfg):
    # Public scalar controls to override with Isaac Lab config overrides.
    wind_enable: bool = True
    wind_min_mps: float = 0.0
    wind_max_mps: float = 2.0  # keep small initially; raise gradually after baseline checks
    wind_calm_probability: float = 0.2
    wind_resample_s: float = 6.0
    wind_ramp_s: float = 1.0
    wind_rho: float = 1.225
    wind_cd_area: float = 0.43
    # Optional max force safety guard for debugging (N); does not correspond to physical saturation.
    wind_max_force_n: float = 60.0  # legacy torso mode. Set <= 0 to remove cap.
    wind_model: str = "torso"  # torso=legacy; panels=per-body box-pressure proxy
    wind_field: str = "uniform"  # uniform, gust, stochastic_gust, vortex
    wind_fixed_heading_deg: float = -1.0  # negative = random heading
    wind_gust_amplitude_mps: float = 0.0
    wind_gust_period_s: float = 3.0
    wind_gust_sigma_fraction: float = 0.15  # synthetic stationary OU turbulence (fraction of local mean speed)
    wind_gust_tau_s: float = 0.8  # autocorrelation time [s], not an empirical hurricane constant
    wind_vortex_core_radius_m: float = 4.0
    wind_vortex_inflow_fraction: float = 0.1
    wind_pressure_coefficient: float = 1.0
    wind_reference_height_m: float = 10.0  # NHC sustained wind classification at 10 m
    wind_shear_exponent: float = 0.0  # 0 = no shear (legacy / benchmark). Storm preset chooses 0.14.
    wind_shear_min_height_m: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        # Direct construction skips Isaac Lab CLI preset resolution. Select the
        # concrete Newton MJWarp preset used by stock G1-Flat playback.
        self.sim.physics = self.sim.physics.newton_mjwarp
        self.compute_final_obs = True  # needed to bootstrap on time-limit truncation with SAC
        # Crucial: intentionally preserve the stock G1 robot, default physics backend,
        # actor observation group, action ordering, and base PPO gait rewards.
        # Remove other base disturbances to isolate the wind effect during initial experiments.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # Change only command distribution (not observation definition).
        # Training on yaw+forward trajectories still uses the stock 3-component command observation.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

