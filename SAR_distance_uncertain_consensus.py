"""
Final revised Robotarium script (ready-to-run).
- Mock Search-and-Rescue scenario with targets & obstacles
- ENV_SCALE to emulate larger arenas while running in Robotarium
- Live overlay of targets & obstacles inside the Robotarium figure (converted to Robotarium coords)
- CSV logging: robot positions, algebraic connectivity, mission metrics
- Robust Wasserstein/Hellinger calculations with stable sqrtm handling
- Uses TkAgg backend and catches WinError 10038 during plt.pause()
"""

import os
import csv
import numpy as np
import random

# Force a robust interactive backend on Windows (must be set before pyplot import)
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.linalg import sqrtm, eigvalsh
import scipy.linalg as la

# Robotarium imports (assumes rps package is installed)
import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

# ---------------- Configuration (edit as needed) ----------------
np.random.seed(100)

N = 20                      # number of robots (reduce if Robotarium limit)
iterations = 800            # simulation steps
std_position = 0.1         # localization noise (std dev in meters)
base_sensing_range = 1    # sensing range at ENV_SCALE = 1
ENV_SCALE = 3.0              # 1.0 -> Robotarium-sized, >1 -> larger mock environment (stretches positions)
mock_search_and_rescue = True
num_targets = 8
target_detection_radius = 0.12  # detection radius for a robot to find a target (meters)
num_obstacles = 5
obstacle_radius = 0.4
sim_in_real_time = True
show_figure = True           # set False to disable live overlays (useful for headless runs)
output_dir = "results_revised"
os.makedirs(output_dir, exist_ok=True)
# ----------------------------------------------------------------

# Instantiate Robotarium
r = robotarium.Robotarium(number_of_robots=N, show_figure=show_figure, sim_in_real_time=sim_in_real_time)

# barrier cert + mapping
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# helper: safe sqrtm that returns real matrix and handles small numerical negatives
def real_sqrtm(mat):
    S = sqrtm(mat)
    if np.iscomplexobj(S):
        S = S.real
    return S

# --- Noise & covariance utilities ---
def add_gaussian_noise(values, mean=0.0, std_dev=0.1):
    """Add zero-mean Gaussian noise to a 2D vector (or array).
       Returns noisy values and isotropic covariance (2x2).
    """
    values = np.asarray(values)
    noise = np.random.normal(mean, std_dev, values.shape)
    noisy = values + noise
    cov = (std_dev**2) * np.eye(values.shape[0])
    return noisy, cov



# --- SAR scenario: generate targets and obstacles in scaled workspace ---
world_min, world_max = -1.0 * ENV_SCALE, 1.0 * ENV_SCALE

if mock_search_and_rescue:
    # --- Fixed targets (8 total, 2 per quadrant) ---
    target_x = [ 0.7,  0.4,   # top-right
                -0.7, -0.4,   # top-left
                 0.7,  0.4,   # bottom-right
                -0.7, -0.4]   # bottom-left

    target_y = [ 0.7,  0.4,   # top-right
                 0.7,  0.4,   # top-left
                -0.7, -0.4,   # bottom-right
                -0.7, -0.4]   # bottom-left

    targets = np.array([target_x, target_y])
    num_targets = len(target_x)
    target_found = np.zeros(num_targets, dtype=bool)

    # --- Fixed obstacles (8 total, 2 per quadrant) ---
    obstacle_x = [ 1.0,  2.2,   # top-right
                  -1.0, -2.2,   # top-left
                   1.0,  2.2,   # bottom-right
                  -1.0, -2.2]   # bottom-left

    obstacle_y = [ 2.2,  1.0,   # top-right
                   2.2,  1.0,   # top-left
                  -2.2, -1.0,   # bottom-right
                  -2.2, -1.0]   # bottom-left

    obstacles = np.array([obstacle_x, obstacle_y])
    num_obstacles = len(obstacle_x)

else:
    targets = np.empty((2,0))
    target_found = np.zeros(0, dtype=bool)
    obstacles = np.empty((2,0))




# ----------------- Logging / results -----------------
positions = {i: {'x': [], 'y': []} for i in range(N)}
algebraic_connectivity_values = []
detection_rates = []

pos_csv = os.path.join(output_dir, "robot_positions.csv")
conn_csv = os.path.join(output_dir, "algebraic_connectivity.csv")
metrics_csv = os.path.join(output_dir, "mission_metrics.csv")

with open(pos_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "robot", "x", "y"])

with open(conn_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "algebraic_connectivity"])

with open(metrics_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "targets_found", "targets_total", "detection_fraction"])

# --- adjacency & laplacian utils (accept covariance arrays) ---
def compute_adjacency_matrix(positions_noisy, covariances, sensing_range):
    Nloc = positions_noisy.shape[1]
    A = np.zeros((Nloc, Nloc))
    W = np.zeros((Nloc, Nloc))
    for i in range(Nloc):
        for j in range(i+1, Nloc):
            #d_ij = wasserstein_distance(positions_noisy[:, i], covariances[:, :, i],
                                        #positions_noisy[:, j], covariances[:, :, j])
            d_ij = 1.3*np.linalg.norm(positions_noisy[:, j] - positions_noisy[:, i])/ENV_SCALE
            if d_ij < sensing_range:
                denom = (sensing_range**2 - d_ij**2)
                w_ij = (2 * sensing_range**2) / (denom**2) if denom >= 1e-6 else 1e6
                A[i, j] = 1
                A[j, i] = 1
                W[i, j] = W[j, i] = w_ij
    return A, W

def compute_degree_matrix(A):
    return np.diag(np.sum(A, axis=1))

def compute_laplacian_matrix(A):
    D = compute_degree_matrix(A)
    return D - A

def algebraic_connectivity(L):
    if L.shape[0] < 2:
        return 0.0
    eigenvals = eigvalsh(L)
    return float(eigenvals[1]) if len(eigenvals) >= 2 else 0.0

# --- obstacle avoidance (velocity repulsion) ---
def apply_obstacle_avoidance(vel, pos_noisy, obstacles, obs_radius, gain=2):
    for i in range(pos_noisy.shape[1]):
        p = pos_noisy[:, i]
        for o in range(obstacles.shape[1]):
            o_pos = obstacles[:, o]
            d = np.linalg.norm(p - o_pos)
            if d < obs_radius + 0.5:
                if d > 1e-6:
                    dir_away = (p - o_pos) / d
                else:
                    dir_away = np.random.randn(2); dir_away /= np.linalg.norm(dir_away)
                strength = gain * (obs_radius + 0.5 - d)
                vel[:, i] += strength * dir_away
    return vel

# ---------------- Robotarium live overlay setup ----------------
plt.ion()
targets_r = (targets / ENV_SCALE) if targets.size else np.empty((2,0))
obstacles_r = (obstacles / ENV_SCALE) if obstacles.size else np.empty((2,0))

t_scat = None
obs_patches = []
robots_overlay = None

if show_figure:
    fig = plt.gcf()
    ax = plt.gca()

    if targets_r.size:
        t_scat = ax.scatter(targets_r[0, :], targets_r[1, :], marker='*', s=120, zorder=4)

    for o in range(obstacles_r.shape[1]):
        c = Circle((obstacles_r[0, o], obstacles_r[1, o]), obstacle_radius / ENV_SCALE,
                   color='gray', alpha=0.4, zorder=1)
        ax.add_patch(c)
        obs_patches.append(c)

    robots_overlay = ax.scatter([], [], s=40, facecolors='none', edgecolors='k', zorder=5)

# ---------------- Main simulation loop ----------------
for k in range(iterations):
    x = r.get_poses()                    # unicycle poses 3xN
    x_si = uni_to_si_states(x)           # single-integrator 2xN (Robotarium coords)
    x_world = x_si * ENV_SCALE           # interpret stretched world coordinates

    # add noise and covariances
    x_si_noisy = np.zeros_like(x_si)
    cov = np.zeros((2, 2, N))
    for i in range(N):
        noisy, cov_i = add_gaussian_noise(x_world[:, i], 0.0, std_position)
        x_si_noisy[:, i] = noisy
        cov[:, :, i] = cov_i

    si_velocities = np.zeros((2, N))
    sensing_range = base_sensing_range * ENV_SCALE
    A, W = compute_adjacency_matrix(x_si_noisy, cov, sensing_range)

    for i in range(N):
        neighbors = np.where(A[i, :] > 0)[0]
        deg = max(len(neighbors), 1)
        for j in neighbors:
            w_ij = W[i, j] / deg if W[i, j] > 0 else 1.0 / deg
            si_velocities[:, i] += 0.5 * w_ij * (x_si_noisy[:, j] - x_si_noisy[:, i])

    if mock_search_and_rescue and obstacles.shape[1] > 0:
        si_velocities = apply_obstacle_avoidance(si_velocities, x_si_noisy, obstacles, obstacle_radius * ENV_SCALE)

    # Map velocities back to Robotarium coords for barrier certificate
    si_velocities_robotarium = si_velocities / ENV_SCALE
    si_velocities_safe = si_barrier_cert(si_velocities_robotarium, x_si)

    dxu = si_to_uni_dyn(si_velocities_safe, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Laplacian & connectivity
    Lmat = compute_laplacian_matrix(A)
    lambda2 = algebraic_connectivity(Lmat)
    algebraic_connectivity_values.append(lambda2)

    # SAR detection metric (using noisy world coords)
    if mock_search_and_rescue and targets.shape[1] > 0:
        for t in range(targets.shape[1]):
            if not target_found[t]:
                dists = np.linalg.norm(x_si_noisy - targets[:, t][:, None], axis=0)
                if np.any(dists <= target_detection_radius * ENV_SCALE):
                    target_found[t] = True

    detection_fraction = float(np.sum(target_found) / max(1, targets.shape[1]))
    detection_rates.append(detection_fraction)

    # logging positions (store world coords)
    with open(pos_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for i in range(N):
            writer.writerow([k, i, x_world[0, i], x_world[1, i]])
            positions[i]['x'].append(x_world[0, i])
            positions[i]['y'].append(x_world[1, i])

    with open(conn_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([k, lambda2])

    with open(metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([k, int(np.sum(target_found)), targets.shape[1], detection_fraction])

    if k % 50 == 0:
        print(f"iter {k}: lambda2={lambda2:.4f}, detection={detection_fraction:.2f}")

    # --- Update live overlay (Robotarium axes) ---
    if show_figure:
        if targets_r.size and t_scat is not None:
            colors = ['green' if target_found[t] else 'red' for t in range(targets_r.shape[1])]
            t_scat.set_color(colors)

        if robots_overlay is not None:
            robots_overlay.set_offsets(np.vstack((x_si[0, :], x_si[1, :])).T)

        # small pause to ensure redraw (defensive against WinError 10038)
        try:
            plt.pause(0.001)
        except OSError:
            # Ignore WinError 10038 (socket issue in some Qt backends on Windows)
            pass

# End sim, call Robotarium end hook
r.call_at_scripts_end()


# ---------------- Post-processing plots ----------------
# Increase fonts globally
plt.rcParams.update({
    "font.size": 19,          # base font size
    "axes.titlesize": 19,     # title
    "axes.labelsize": 19,     # x/y labels
    "xtick.labelsize": 19,    # x ticks
    "ytick.labelsize": 19,    # y ticks
    "legend.fontsize": 19     # legend
})

plt.ioff()

# ---------------- Plot: Summary metrics ----------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(len(algebraic_connectivity_values)), algebraic_connectivity_values, label='λ₂')
plt.xlabel('Iteration')
plt.ylabel('Algebraic connectivity')
plt.grid(True)
plt.title('Algebraic connectivity over time')

plt.subplot(1, 2, 2)
plt.plot(range(len(detection_rates)), detection_rates, label='Detection fraction')
plt.xlabel('Iteration')
plt.ylabel('Target Detected')
plt.grid(True)
plt.title('Target Detection over time')

plt.tight_layout()
# Save as PDF
plt.savefig(os.path.join(output_dir, "summary_plots.pdf"))
plt.show()

# ---------------- Plot: Robot trajectories ----------------
plt.figure(figsize=(8, 6))

# HSV colormap for N robots
colors = plt.cm.hsv(np.linspace(0, 1, N))

for i in range(N):
    plt.plot(positions[i]['x'], positions[i]['y'], color=colors[i], linewidth=1.5)

# Plot targets and obstacles
if targets.size:
    plt.scatter(targets[0, :], targets[1, :], marker='*', s=180, label='Targets', color='gold')
if obstacles.size:
    plt.scatter(obstacles[0, :], obstacles[1, :], marker='o', s=150, label='Obstacles', color='red')

plt.title('Robot trajectories')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "trajectories.pdf"))
plt.show()

print("Simulation finished. Results saved in:", output_dir)
