"""Optional Isaac Lab 3.0 native MP4 recording for Newton MJWarp runs.

Do not use Gymnasium RecordVideo here: the stock v3.0 manager env owns a
VideoRecorder, and the custom wind step already calls recorder.step().
"""
from pathlib import Path


def configure_video(cfg, *, output_dir: Path, video_length: int = 200,
                    video_interval: int = 0, headless: bool = False,
                    fps: int = 50, filename_prefix: str = 'g1_wind') -> None:
    """Attach one Newton GL visualizer and one native video recorder to cfg.

    video_length/video_interval are VECTOR ENV STEPS, not aggregate transitions.
    With stock G1 Flat step_dt=0.02 s, 200 steps is 4 simulated seconds.
    """
    if video_length < 1 or video_interval < 0 or fps < 1:
        raise ValueError('video_length and fps must be positive; video_interval must be >= 0')
    if video_interval and video_interval < video_length:
        raise ValueError('video_interval must be 0 (one clip) or >= video_length')

    # Import only when recording is explicitly enabled. This keeps headless SAC
    # runs free of rendering/video dependencies.
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            headless=headless,
            window_width=1280,
            window_height=720,
            streaming_view=False,
            # This initial world-space view is overridden at each recorded step
            # by follow_robot_camera() using sim.set_camera_view().
            # Kit-only origin_type/origin_track_path are intentionally omitted.
            eye=(3.2, -3.6, 2.1),
            lookat=(0.0, 0.0, 0.85),
            max_visible_envs=1,
            visible_env_indices=[0],
            randomly_sample_visible_envs=False,
        )
    ]
    cfg.video_recorders = [
        VideoRecorderCfg(
            source='visualizer:newton_gl',
            output_dir=str(output_dir),
            output_filename_prefix=filename_prefix,
            video_length=video_length,
            video_interval=video_interval,
            fps=fps,
        )
    ]


# Camera position relative to the root of the first G1; meters in world XYZ.
_CAMERA_EYE_OFFSET = (2.6, -3.4, 1.7)
_CAMERA_TARGET_OFFSET = (0.0, 0.0, 0.25)


def follow_robot_camera(env, env_index: int = 0) -> None:
    """Point the Newton GL viewport camera at a moving G1 robot.

    G1 root_pos_w is world-space, including the per-environment origin offset.
    This avoids the fixed (0,0,0) camera missing env_0 and does not require
    auto-created cameras (which can fail with replicated Newton worlds).

    Call *before* env.step() when capturing video.  The one-step camera lag is
    at most one control step, and only 3 values are transferred to the CPU.
    """
    root_xyz = env.scene["robot"].data.root_pos_w[env_index].detach().cpu().tolist()
    eye = tuple(float(p + d) for p, d in zip(root_xyz, _CAMERA_EYE_OFFSET))
    target = tuple(float(p + d) for p, d in zip(root_xyz, _CAMERA_TARGET_OFFSET))
    env.sim.set_camera_view(eye, target)


def camera_capture_active(step_zero_based: int, length: int, interval: int) -> bool:
    """Only move the camera when a video clip is being captured.

    Native recorder's first clip starts at environment step 1; subsequent clips
    begin every `interval` steps. Inputs here are Python loop (0-based) steps.
    """
    if length <= 0 or interval < 0 or (interval and interval < length):
        raise ValueError("Invalid video recording window")
    return step_zero_based < length if interval == 0 else (step_zero_based % interval) < length
