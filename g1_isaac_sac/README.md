# Wind-Resistant Unitree G1 — Residual SAC

This project investigates **wind-resistant humanoid locomotion** in simulation using the **Unitree G1** and **Soft Actor-Critic (SAC)**. It is part of a broader study of reinforcement learning and model predictive control for maintaining balance under extreme wind disturbances.

Built with **Isaac Lab 3.0 (Newton MJWarp)** and **PyTorch**, the controller combines a **frozen pretrained G1 walking policy** with a trainable **residual SAC policy**. SAC learns corrections to the nominal joint actions rather than learning walking from scratch.

The implementation includes synthetic gusting wind and distributed aerodynamic loading, an adaptive wind curriculum, training and checkpointing, and evaluations comparing the frozen walking policy with the SAC-augmented controller.

## Main files

- `train_sac.py` — Train the residual SAC controller.
- `evaluate_paired_audit.py` — Compare the frozen policy and SAC controller under matched wind conditions.
- `evaluate_pretrained_wind.py` — Evaluate the pretrained walking baseline.
- `play_sac.py` — Play back a saved SAC controller.
- `calibrate_aero.py` — Inspect aerodynamic model parameters.
- `g1_wind/` — Environment, wind model, SAC implementation, and utilities.
- `tests/` — CPU-level checks for core components.

## Demo

[Latest training video](media/train_0039.mp4)

## Status

This is a **research prototype**. The included video shows a training rollout, not a verified improvement in wind resistance. Comparative evaluations against the frozen baseline and simulator-level validation of aerodynamic forces remain necessary.

**Dependencies:** Isaac Lab 3.0, its Newton MJWarp backend, PyTorch, and the NVIDIA pretrained G1-Flat walking checkpoint. Pretrained weights and training checkpoints are not included in this repository.
