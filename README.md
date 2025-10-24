# uncertain-consensus

**Uncertain Consensus — Probabilistic Consensus with Distributional Distances (Robotarium)**

A lightweight Robotarium-based implementation of a consensus algorithm that accounts for uncertainty in robot state estimates. The project experiments with distributional distances (Wasserstein, Hellinger) and covariance-aware weighting for consensus updates, combined with Control Barrier Functions (CBF) to enforce safety in single-integrator → unicycle transformations.

---

## Repository name

`uncertain-consensus`

---

## Summary

This repository contains a single-file implementation (Python) that runs a multi-robot consensus experiment on the Robotarium simulator. Robots share noisy position estimates and per-robot covariance information; the consensus weights between neighbors are computed using distributional distances (e.g., 2‑Wasserstein), and a barrier certificate is used to guarantee collision avoidance. The script also records robot positions and a smoothed convergence metric for later analysis.

---

## Key features

* Noise injection on robot poses and per-robot covariance generation.
* Distance metrics between uncertain estimates: 2‑Wasserstein and Hellinger implementations.
* Covariance-aware consensus weights (distance-to-weight mapping).
* Single‑integrator barrier certificate for collision avoidance and safe unicycle control mapping.
* Logging of robot trajectories (`robot_positions.csv`) and performance metrics (`performance_metrics.csv`).
* Simple plotting of trajectories and smoothed convergence metric at the end of the run.

---

## Requirements

* Python 3.8+
* NumPy, SciPy
* Matplotlib
* Robotarium Python utilities (`rps`) — Robotarium installation and configuration required.

Install example:

```bash
pip install numpy scipy matplotlib
# Install Robotarium / rps according to your Robotarium environment
```

---

## Quick start

1. Make sure Robotarium (`rps`) is installed and reachable from Python.
2. Place `uncertain-consensus.py` in the repository root (rename from `uncertain-consensus (1).py` if needed).
3. Run:

```bash
python uncertain-consensus.py
```

The script will open the Robotarium figure, run the configured number of iterations, save CSV logs, and show trajectory/performance plots.

---

## Configuration

Most experiment parameters are defined near the top of the script:

* `N` — number of robots
* `iterations` — simulation steps
* `std_position` — standard deviation used to inject Gaussian noise
* `sensing_range` — neighbor detection range used for delta-disk neighbor calculation
* Choice of distance metric: `wasserstein_distance`, `hellinger_distance`, `max_distance`, or `min_distance` (comment/uncomment in main loop)

Adjust these to match experiment goals and Robotarium scale.

---

## Outputs

* `robot_positions.csv` — per-iteration X/Y positions for each robot
* `performance_metrics.csv` — smoothed convergence metric over time
* Matplotlib figures showing trajectories and convergence

---

## Notes & suggestions

* Current covariance generation is derived from injected noise and may be improved with a formal estimator (EKF/UKF) if you want more realistic covariance evolution.
* The mapping from distributional distance to consensus weight is provided in the code; experiment with different weighting functions or normalization for stability.
* If you plan to run on a physical testbed, re-check the barrier certificate parameters, maximum speeds, and robot dynamics mapping.

---

## License

Add a `LICENSE` file (MIT recommended) if you want to make this public.

---

If you want, I can also:

* create `requirements.txt` and `run.sh`;
* rename the uploaded file and update the README with exact file names;
* add an MIT `LICENSE` file;
* convert the code into a small package with clearer entry points.

Which would you like next?
