"""Video setup contract tests without needing Isaac Lab/visualization at import time."""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from g1_wind_sac_v5.recording import configure_video, follow_robot_camera, camera_capture_active


@pytest.mark.parametrize('length,interval,fps', [(0, 0, 50), (10, 9, 50), (10, -1, 50), (10, 0, 0)])
def test_bad_video_parameters_fail_before_import(length, interval, fps):
    with pytest.raises(ValueError):
        configure_video(SimpleNamespace(), output_dir=Path('/tmp/videos'),
                        video_length=length, video_interval=interval, fps=fps)


def test_native_newton_recorder_is_configured(monkeypatch, tmp_path):
    def register(name):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    register('isaaclab')
    register('isaaclab.envs')
    register('isaaclab.envs.utils')
    video_module = register('isaaclab.envs.utils.video_recorder_cfg')
    register('isaaclab_visualizers')
    newton_module = register('isaaclab_visualizers.newton')

    class VideoRecorderCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class NewtonGLVisualizerCfg:
        # Model the supported v3.0 arguments relevant to this project.
        # Intentionally NO Kit-specific camera tracking args.
        def __init__(self, *, headless=False, window_width=1920,
                     window_height=1080, streaming_view=True,
                     eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0),
                     max_visible_envs=None, visible_env_indices=None,
                     randomly_sample_visible_envs=True):
            self.kwargs = locals().copy()

    video_module.VideoRecorderCfg = VideoRecorderCfg
    newton_module.NewtonGLVisualizerCfg = NewtonGLVisualizerCfg
    cfg = SimpleNamespace(sim=SimpleNamespace(visualizer_cfgs=[]), video_recorders=[])
    configure_video(cfg, output_dir=tmp_path / 'clips', video_length=200,
                    video_interval=1000, fps=50, headless=True)
    assert len(cfg.sim.visualizer_cfgs) == 1
    assert cfg.sim.visualizer_cfgs[0].kwargs['eye'] == (3.2, -3.6, 2.1)
    assert cfg.sim.visualizer_cfgs[0].kwargs['headless'] is True
    assert cfg.sim.visualizer_cfgs[0].kwargs['visible_env_indices'] == [0]
    assert cfg.video_recorders[0].kwargs['source'] == 'visualizer:newton_gl'
    assert cfg.video_recorders[0].kwargs['video_length'] == 200
    assert cfg.video_recorders[0].kwargs['video_interval'] == 1000


def test_camera_tracks_robot_world_position():
    import torch
    from types import SimpleNamespace
    called = []
    sim = SimpleNamespace(set_camera_view=lambda eye, target: called.append((eye, target)))
    root_xyz = torch.tensor([[12.0, -9.0, 0.9], [10.0, 2.0, 1.0]])
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=root_xyz))}, sim=sim)
    follow_robot_camera(env)
    eye, target = called[0]
    assert eye == pytest.approx((14.6, -12.4, 2.6))
    assert target == pytest.approx((12.0, -9.0, 1.15))
    follow_robot_camera(env, env_index=1)
    assert called[1][1] == pytest.approx((10.0, 2.0, 1.25))


def test_follow_camera_only_active_during_recorded_clips():
    assert all(camera_capture_active(s, 200, 1000) for s in range(200))
    assert not camera_capture_active(200, 200, 1000)
    assert not camera_capture_active(999, 200, 1000)
    assert camera_capture_active(1000, 200, 1000)
    assert not camera_capture_active(1200, 200, 1000)
    assert camera_capture_active(0, 200, 0)
    assert not camera_capture_active(200, 200, 0)
