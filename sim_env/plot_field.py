"""Render reproducible wind-map illustrations with matplotlib.

Run from the project directory: python -m sim_env.plot_field
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .config import DEFAULT_TORNADO
from .wind import local_to_world, tornado_velocity


def _field_grid(x, y, config, scale):
    xx, yy = np.meshgrid(x, y)
    velocities = tornado_velocity(np.stack((xx, yy), axis=-1), config, scale)
    return xx, yy, velocities, np.linalg.norm(velocities, axis=-1)


def _save(fig, output_dir, stem, dpi=240):
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        paths.append(path)
    plt.close(fig)
    return paths


def render_black_white_vectors(output_dir=None, config=None, scale=1.0):
    """Proposal figure: black arrows on white, length proportional to speed.

    Uses the same physical field as the simulator, without normalizing its
    vectors. A fixed display scale maps 3.5 m/s to 1 m on the map. The zero
    vector at the vortex center is omitted rather than drawn as a dot.
    """
    cfg = DEFAULT_TORNADO if config is None else config
    destination = Path(__file__).resolve().parent / "assets" if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    spacing = cfg.map_size_m / 25
    grid = np.linspace(spacing / 2, cfg.map_size_m - spacing / 2, 25)
    xx, yy, velocity, speed = _field_grid(grid, grid, cfg, scale)
    nonzero = speed > 1e-12
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.quiver(xx[nonzero], yy[nonzero], velocity[..., 0][nonzero], velocity[..., 1][nonzero],
                  color="black", angles="xy", scale_units="xy", scale=3.5,
                  pivot="mid", units="width", width=.0018, headwidth=3.3,
                  headlength=4.5, headaxislength=4.0, minlength=0, minshaft=1)
        ax.set(xlim=(0, cfg.map_size_m), ylim=(0, cfg.map_size_m), aspect="equal")
        ax.set_axis_off()
        return _save(fig, destination, "tornado_vector_field_bw", dpi=300)


def render_maps(output_dir=None, config=None, scale=1.0):
    cfg = DEFAULT_TORNADO if config is None else config
    destination = (Path(__file__).resolve().parent / "assets"
                   if output_dir is None else Path(output_dir))
    destination.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.titlesize": 17, "axes.labelsize": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "figure.facecolor": "#f7f9fc", "axes.facecolor": "#ffffff",
             "text.color": "#122238", "axes.labelcolor": "#122238",
             "xtick.color": "#31475d", "ytick.color": "#31475d"}
    outputs = []
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10.6, 9.1))
        fig.subplots_adjust(left=0.09, right=0.85, top=0.86, bottom=0.12)
        grid = np.linspace(0, cfg.map_size_m, 401)
        xx, yy, wind, speed = _field_grid(grid, grid, cfg, scale)
        maximum = scale * np.hypot(cfg.peak_tangential_speed_mps,
                                   cfg.peak_inflow_speed_mps)
        vmax = max(float(maximum), 1e-6)
        shade = ax.pcolormesh(xx, yy, speed, shading="auto", cmap="viridis",
                             vmin=0, vmax=vmax, rasterized=True)
        arrows = np.linspace(2.5, cfg.map_size_m - 2.5, 21)
        qx, qy, qw, _ = _field_grid(arrows, arrows, cfg, scale)
        ax.quiver(qx, qy, qw[..., 0], qw[..., 1], color="white",
                  alpha=0.82, angles="xy", scale_units="xy", scale=3.5,
                  width=0.0025, headwidth=3.6, headlength=4.3)
        ax.add_patch(Circle(cfg.center_m, cfg.core_radius_m, fill=False,
                            edgecolor="white", linewidth=1.3, linestyle=(0, (4, 4)),
                            alpha=0.8))
        ax.plot(*cfg.center_m, marker="+", color="white", markersize=12, mew=2)
        ax.annotate("Vortex center", xy=cfg.center_m, xytext=(29, 31),
                    color="white", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "white", "lw": 1})
        px, py = cfg.patch_origin_m
        ax.add_patch(Rectangle((px, py), cfg.patch_size_m, cfg.patch_size_m,
                               fill=False, edgecolor="#ff6655", linewidth=3))
        ax.annotate("WALKING TEST REGION\n10 m × 10 m", xy=(px+5, py+10),
                    xytext=(73, 73), ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#122238",
                    bbox={"boxstyle": "round,pad=0.6", "fc": "white", "ec": "none"},
                    arrowprops={"arrowstyle": "->", "color": "#ff6655", "lw": 2})
        ax.set(xlim=(0, cfg.map_size_m), ylim=(0, cfg.map_size_m),
               xlabel="World x [m]", ylabel="World y [m]", aspect="equal")
        ax.set_xticks(np.arange(0, cfg.map_size_m + 1, 10))
        ax.set_yticks(np.arange(0, cfg.map_size_m + 1, 10))
        cax = fig.add_axes([0.88, 0.21, 0.025, 0.56])
        colorbar = fig.colorbar(shade, cax=cax)
        colorbar.set_label("Wind speed [m/s]", labelpad=12)
        fig.text(0.09, 0.94, "Idealized tornado wind field", fontsize=23, fontweight="bold")
        fig.text(0.09, 0.905, "100 m × 100 m horizontal slice · prescribed steady velocity field", fontsize=12)
        fig.text(0.09, 0.055,
                 f"Core radius {cfg.core_radius_m:g} m  |  Peak tangential speed {scale*cfg.peak_tangential_speed_mps:g} m/s  |  "
                 f"Peak inflow {scale*cfg.peak_inflow_speed_mps:g} m/s", fontsize=10)
        fig.text(0.09, 0.029, "Synthetic control benchmark. No resolved turbulence, vertical flow, or robot–flow coupling.",
                 fontsize=9, color="#54687d")
        outputs.extend(_save(fig, destination, "tornado_field"))

        fig, ax = plt.subplots(figsize=(10.6, 9.1))
        fig.subplots_adjust(left=0.09, right=0.85, top=0.86, bottom=0.13)
        local = np.linspace(0, cfg.patch_size_m, 201)
        lx, ly = np.meshgrid(local, local)
        position = local_to_world(np.stack((lx, ly), axis=-1), cfg)
        velocity = tornado_velocity(position, cfg, scale)
        local_speed = np.linalg.norm(velocity, axis=-1)
        lower, upper = float(local_speed.min()), float(local_speed.max())
        if upper - lower < 1e-6:
            upper = lower + 1e-6
        shade = ax.pcolormesh(lx, ly, local_speed, shading="auto", cmap="viridis",
                             vmin=lower, vmax=upper, rasterized=True)
        arrow_local = np.linspace(0.4, cfg.patch_size_m - 0.4, 13)
        qx, qy = np.meshgrid(arrow_local, arrow_local)
        qw = tornado_velocity(local_to_world(np.stack((qx, qy), axis=-1), cfg), cfg, scale)
        ax.quiver(qx, qy, qw[..., 0], qw[..., 1], color="white",
                  alpha=0.8, angles="xy", scale_units="xy", scale=15,
                  width=0.003, headwidth=3.4, headlength=4.0)
        ax.plot([1, 5.55], [5, 5], color="#ff6655", lw=3, zorder=5)
        ax.annotate("", xy=(4.7, 5), xytext=(3.6, 5),
                    arrowprops={"arrowstyle": "-|>", "lw": 3, "color": "#ff6655"}, zorder=6)
        ax.scatter([1], [5], s=125, facecolor="white", edgecolor="#122238", linewidth=2, zorder=7)
        ax.scatter([5.55], [5], s=170, marker="*", facecolor="#ff6655", edgecolor="white", linewidth=1, zorder=7)
        label_box = {"boxstyle": "round,pad=0.35", "fc": "white", "ec": "none", "alpha": 0.95}
        ax.text(1, 4.4, "START\n(1, 5)", ha="center", va="top", fontsize=10, bbox=label_box)
        ax.text(5.55, 4.4, "GOAL\n(5.55, 5)", ha="center", va="top", fontsize=10, bbox=label_box)
        ax.text(3.275, 5.5, "Nominal walking segment", ha="center", fontsize=10, bbox=label_box)
        ax.set(xlim=(0, cfg.patch_size_m), ylim=(0, cfg.patch_size_m),
               xlabel="Local x [m]", ylabel="Local y [m]", aspect="equal")
        ax.set_xticks(np.arange(0, cfg.patch_size_m + 1, 1))
        ax.set_yticks(np.arange(0, cfg.patch_size_m + 1, 1))
        cax = fig.add_axes([0.88, 0.21, 0.025, 0.56])
        colorbar = fig.colorbar(shade, cax=cax)
        colorbar.set_label("Local wind speed [m/s]", labelpad=12)
        fig.text(0.09, 0.94, "Walking test region", fontsize=23, fontweight="bold")
        xmin, xmax, ymin, ymax = cfg.patch_bounds_m
        fig.text(0.09, 0.905, f"10 m × 10 m patch · world x ∈ [{xmin:g}, {xmax:g}] m, y ∈ [{ymin:g}, {ymax:g}] m", fontsize=12)
        fig.text(0.09, 0.063, f"World position = local position + ({px:g}, {py:g}) m  |  "
                 f"Wind-speed range {local_speed.min():.2f}–{local_speed.max():.2f} m/s", fontsize=10)
        fig.text(0.09, 0.034, "The route is a nominal test reference, not an optimized path. Wind-force magnitude also depends on robot velocity.",
                 fontsize=9, color="#54687d")
        outputs.extend(_save(fig, destination, "walking_test_region"))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--wind-scale", type=float, default=1.0)
    parser.add_argument("--style", choices=("colored", "arrows-bw"), default="colored",
                        help="arrows-bw: clean black arrows on white for the proposal")
    args = parser.parse_args()
    render = render_black_white_vectors if args.style == "arrows-bw" else render_maps
    for output in render(args.output_dir, scale=args.wind_scale):
        print(output)


if __name__ == "__main__":
    main()
