"""Isaac Lab 3.0 G1 wind-residual-SAC task registration.

Pure-PyTorch SAC modules are also importable for CPU tests without Isaac Lab.
"""
try:
    import gymnasium as gym
except ModuleNotFoundError as exc:
    if exc.name != 'gymnasium':
        raise
else:
    gym.register(
        id='Isaac-Velocity-Flat-G1-Wind-SAC-V5-5090',
        entry_point='g1_wind_sac_v5.env:G1WindSACEnv',
        disable_env_checker=True,
        kwargs={'env_cfg_entry_point': 'g1_wind_sac_v5.cfg:G1WindSACEnvCfg'},
    )
