"""
Final revised Robotarium script (ready-to-run).
- Mock Search-and-Rescue scenario with targets & obstacles
- ENV_SCALE to emulate larger arenas while running in Robotarium
- Live overlay of targets & obstacles inside the Robotarium figure (converted to Robotarium coords)
- CSV logging: robot positions, algebraic connectivity, mission metrics, Wasserstein summaries,
  AND obstacle-avoidance metrics (per-iteration)
- Robust Wasserstein/Hellinger calculations with stable sqrtm handling
- Computes Wasserstein distances between robot Gaussians and target/obstacle Gaussians
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
base_sensing_range = 1     # sensing range at ENV_SCALE = 1
ENV_SCALE = 3.0              # 1.0 -> Robotarium-sized, >1 -> larger mock environment (stretches positions)
mock_search_and_rescue = True
num_targets = 8
target_detection_radius = 0.2  # detection radius for a robot to find a target (meters)
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
si_to_si_dyn, uni_to_si_states = create_si_to_uni_mapping() if False else (None, None)  # keep backward compat if needed
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

def wasserstein_distance(mean1, cov1, mean2, cov2):
    """2-Wasserstein distance between two Gaussians in R^2. Robust to numeric issues."""
    mean1 = np.asarray(mean1).reshape(-1)
    mean2 = np.asarray(mean2).reshape(-1)
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)
    mean_diff_norm2 = np.sum((mean1 - mean2)**2)
    # robust sqrt handling
    S1 = real_sqrtm(cov1)
    inner = S1 @ cov2 @ S1
    S_inner = real_sqrtm(inner)
    trace_term = np.trace(cov1 + cov2 - 2.0 * S_inner)
    val = mean_diff_norm2 + trace_term
    val = max(val, 0.0)
    return np.sqrt(val)

def hellinger_distance(mu1, cov1, mu2, cov2):
    """Hellinger distance for Gaussians (R^2)."""
    det1 = max(la.det(cov1), 1e-12)
    det2 = max(la.det(cov2), 1e-12)
    cov_sum = (cov1 + cov2) / 2.0
    det_sum = max(la.det(cov_sum), 1e-12)
    diff = mu1 - mu2
    inv_cov_sum = la.inv(cov_sum)
    exponent = -0.125 * (diff.T @ inv_cov_sum @ diff)
    coeff = (det1**0.25 * det2**0.25) / (det_sum**0.5)
    inside = coeff * np.exp(exponent)
    inside = np.clip(inside, 0.0, 1.0)
    return np.sqrt(1.0 - inside)

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
mean_min_w_target_list = []
mean_min_w_obstacle_list = []
num_robots_near_obstacle_list = []
min_distance_to_obstacle_list = []
collision_counts = []

pos_csv = os.path.join(output_dir, "robot_positions.csv")
conn_csv = os.path.join(output_dir, "algebraic_connectivity.csv")
metrics_csv = os.path.join(output_dir, "mission_metrics.csv")
target_wass_csv = os.path.join(output_dir, "target_wasserstein.csv")
obstacle_wass_csv = os.path.join(output_dir, "obstacle_wasserstein.csv")
obstacle_metrics_csv = os.path.join(output_dir, "obstacle_avoidance_metrics.csv")  # NEW

with open(pos_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "robot", "x", "y"])

with open(conn_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "algebraic_connectivity"])

with open(metrics_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "targets_found", "targets_total", "detection_fraction"])

with open(target_wass_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "mean_min_wasserstein_to_target", "median_min_wasserstein_to_target"])

with open(obstacle_wass_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "mean_min_wasserstein_to_obstacle", "median_min_wasserstein_to_obstacle"])

with open(obstacle_metrics_csv, "w", newline="") as f:  # NEW header
    writer = csv.writer(f)
    writer.writerow([
        "iteration",
        "mean_min_w_to_obstacle",
        "median_min_w_to_obstacle",
        "mean_min_dist_to_obstacle",
        "median_min_dist_to_obstacle",
        "num_robots_within_avoidance_threshold",
        "num_collisions"
    ])

# --- adjacency & laplacian utils (accept covariance arrays) ---
def compute_adjacency_matrix(positions_noisy, covariances, sensing_range):
    Nloc = positions_noisy.shape[1]
    A = np.zeros((Nloc, Nloc))
    W = np.zeros((Nloc, Nloc))
    for i in range(Nloc):
        for j in range(i+1, Nloc):
            d_ij = 1.3*wasserstein_distance(positions_noisy[:, i], covariances[:, :, i],
                                        positions_noisy[:, j], covariances[:, :, j])/ENV_SCALE
            #d_ij = np.linalg.norm(positions_noisy[:, j] - positions_noisy[:, i])
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

# --- compute pairwise Wasserstein matrices between robots and targets/obstacles ---
def build_point_set_covariances(points, base_radius):
    """
    Create isotropic covariance matrices for a set of 2D points.
    base_radius (float): a length scale (meters) used to create covariance ~ (base_radius)^2 * I
    Returns covs shape (2,2,M)
    """
    M = points.shape[1] if points.size else 0
    covs = np.zeros((2, 2, M))
    for m in range(M):
        sigma = max(base_radius, 1e-3)
        covs[:, :, m] = (sigma**2) * np.eye(2)
    return covs

def compute_pairwise_wasserstein(robots_means, robots_covs, other_means, other_covs):
    """
    robots_means: (2, N)
    robots_covs: (2,2,N)
    other_means: (2, M)
    other_covs: (2,2,M)
    returns matrix D of shape (N, M) where D[i,m] = W2(robot_i, other_m)
    """
    if other_means.size == 0:
        return np.empty((robots_means.shape[1], 0))
    Nloc = robots_means.shape[1]
    M = other_means.shape[1]
    D = np.zeros((Nloc, M))
    for i in range(Nloc):
        mu_r = robots_means[:, i]
        cov_r = robots_covs[:, :, i]
        for m in range(M):
            mu_o = other_means[:, m]
            cov_o = other_covs[:, :, m]
            D[i, m] = wasserstein_distance(mu_r, cov_r, mu_o, cov_o)
    return D

def pairwise_wasserstein(meansA, covsA, meansB, covsB):
    """
    Compute pairwise 2-Wasserstein distances between two sets of Gaussians.
    meansA: (2, NA), covsA: (2,2,NA)
    meansB: (2, NB), covsB: (2,2,NB)
    Returns distances: shape (NA, NB) where d[i,j] = W( A_i, B_j )
    """
    NA = meansA.shape[1]
    NB = meansB.shape[1]
    dists = np.zeros((NA, NB))
    for i in range(NA):
        for j in range(NB):
            dists[i, j] = wasserstein_distance(meansA[:, i], covsA[:, :, i],
                                               meansB[:, j], covsB[:, :, j])
    return dists

# --- obstacle avoidance (velocity repulsion) ---
def apply_obstacle_avoidance(vel, pos_noisy, covariances, obstacles_means, obstacles_covs,
                             obs_radius, gain=2, margin=0.5):
    """
    Wasserstein-based obstacle avoidance:
    - Uses Wasserstein distance d for gating/strength
    - Direction = normalized Euclidean mean-difference (stable repulsion)
    """
    Nloc = pos_noisy.shape[1]
    M = obstacles_means.shape[1]
    if M == 0:
        return vel

    for i in range(Nloc):
        m_r = pos_noisy[:, i]
        C_r = covariances[:, :, i]
        for o in range(M):
            m_o = obstacles_means[:, o]
            C_o = obstacles_covs[:, :, o]

            # Wasserstein distance between Gaussians
            d = wasserstein_distance(m_r, C_r, m_o, C_o)

            if d < (obs_radius + margin):
                diff = m_r - m_o
                norm = np.linalg.norm(diff)
                if norm < 1e-1:
                    # fallback direction
                    diff = np.random.randn(2)
                    norm = np.linalg.norm(diff)
                dir_unit = diff / norm

                # stronger repulsion if closer
                strength = gain * (obs_radius + margin - d)
                vel[:, i] += strength * dir_unit

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
        t_scat = ax.scatter(targets_r[0, :], targets_r[1, :], marker='*', s=120, zorder=4, label='Targets')

    for o in range(obstacles_r.shape[1]):
        c = Circle((obstacles_r[0, o], obstacles_r[1, o]), obstacle_radius / ENV_SCALE,
                   color='gray', alpha=0.4, zorder=1)
        ax.add_patch(c)
        obs_patches.append(c)
    # For obstacles legend entry, add a dummy handle if obstacles exist
    if obstacles_r.size:
        ax.scatter([], [], s=150, marker='o', color='red', alpha=0.6, label='Obstacles')

    robots_overlay = ax.scatter([], [], s=40, facecolors='none', edgecolors='k', zorder=5)

# ---------------- Precompute covariances for targets & obstacles (in world coords) ----------------
# Choose covariances so that obstacles are "wider" than targets
if targets.size:
    # Targets small — reduced base radius
    target_base_radius = max(0.6* target_detection_radius * ENV_SCALE, 1e-3)
    target_covs = build_point_set_covariances(targets, target_base_radius)
else:
    target_covs = np.empty((2,2,0))

if obstacles.size:
    # Obstacles: moderate size but not enormous
    obstacle_base_radius = max(0.6 * obstacle_radius * ENV_SCALE, 1e-3)
    obstacle_covs = build_point_set_covariances(obstacles, obstacle_base_radius)
else:
    obstacle_covs = np.empty((2,2,0))

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

    # --- compute Wasserstein distances to targets / obstacles this iteration ---
    if targets.size:
        W_robot_target = compute_pairwise_wasserstein(x_si_noisy, cov, targets, target_covs)
        # per-robot minimum distance to any target
        min_w_to_target = np.min(W_robot_target, axis=1) if W_robot_target.size else np.array([])
        mean_min_w_to_target = float(np.mean(min_w_to_target)) if min_w_to_target.size else np.nan
        median_min_w_to_target = float(np.median(min_w_to_target)) if min_w_to_target.size else np.nan
    else:
        W_robot_target = np.empty((N, 0))
        min_w_to_target = np.array([])
        mean_min_w_to_target = np.nan
        median_min_w_to_target = np.nan

    if obstacles.size:
        W_robot_obstacle = compute_pairwise_wasserstein(x_si_noisy, cov, obstacles, obstacle_covs)
        min_w_to_obstacle = np.min(W_robot_obstacle, axis=1) if W_robot_obstacle.size else np.array([])
        mean_min_w_to_obstacle = float(np.mean(min_w_to_obstacle)) if min_w_to_obstacle.size else np.nan
        median_min_w_to_obstacle = float(np.median(min_w_to_obstacle)) if min_w_to_obstacle.size else np.nan
    else:
        W_robot_obstacle = np.empty((N, 0))
        min_w_to_obstacle = np.array([])
        mean_min_w_to_obstacle = np.nan
        median_min_w_to_obstacle = np.nan

    # Log Wasserstein summaries
    with open(target_wass_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([k, mean_min_w_to_target, median_min_w_to_target])
    with open(obstacle_wass_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([k, mean_min_w_to_obstacle, median_min_w_to_obstacle])

    si_velocities = np.zeros((2, N))
    sensing_range = base_sensing_range * ENV_SCALE
    A, W = compute_adjacency_matrix(x_si_noisy, cov, sensing_range)

    for i in range(N):
        neighbors = np.where(A[i, :] > 0)[0]
        deg = max(len(neighbors), 1)
        for j in neighbors:
            w_ij = W[i, j] / deg if W[i, j] > 0 else 1.0 / deg
            si_velocities[:, i] += 0.5 * w_ij * (x_si_noisy[:, j] - x_si_noisy[:, i])

    avoidance_count = 0
    if mock_search_and_rescue and obstacles.shape[1] > 0:
        si_velocities = apply_obstacle_avoidance(si_velocities, x_si_noisy, cov,
                                                 obstacles, obstacle_covs,
                                                 obstacle_radius * ENV_SCALE)

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
        # pairwise distances robot (N) x targets (T)
        d_rt = pairwise_wasserstein(x_si_noisy, cov, targets, target_covs)  # shape (N, T)
        for t in range(targets.shape[1]):
            if not target_found[t]:
                # if any robot's Wasserstein distance to target <= detection threshold, mark found
                if np.any(d_rt[:, t] <= target_detection_radius * ENV_SCALE):
                    target_found[t] = True

    detection_fraction = float(np.sum(target_found) / max(1, targets.shape[1]))
    detection_rates.append(detection_fraction)
    mean_min_w_target_list.append(mean_min_w_to_target)
    mean_min_w_obstacle_list.append(mean_min_w_to_obstacle)

    # ---------------- Obstacle-avoidance derived metrics (NEW) ----------------
    if obstacles.size:
        # compute Euclidean min distance from each robot to closest obstacle (world coords)
        min_dists = np.full(N, np.inf)
        for i in range(N):
            if obstacles.size:
                ds = np.linalg.norm(x_si_noisy[:, i][:, None] - obstacles, axis=0)
                min_dists[i] = float(np.min(ds))
            else:
                min_dists[i] = np.nan

        mean_min_dist = float(np.nanmean(min_dists)) if min_dists.size else np.nan
        median_min_dist = float(np.nanmedian(min_dists)) if min_dists.size else np.nan
        # threshold used in avoidance function (same as repulsion condition)
        avoidance_threshold = obstacle_radius * ENV_SCALE + 0.5
        num_robots_near = int(np.sum(min_dists <= avoidance_threshold))
        # collisions (penetration inside obstacle radius)
        num_collisions = int(np.sum(min_dists <= (obstacle_radius * ENV_SCALE)))
    else:
        mean_min_dist = np.nan
        median_min_dist = np.nan
        num_robots_near = 0
        num_collisions = 0

    num_robots_near_obstacle_list.append(num_robots_near)
    min_distance_to_obstacle_list.append(mean_min_dist)
    collision_counts.append(num_collisions)

    # log positions (store world coords)
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

    # write obstacle metrics csv (NEW)
    with open(obstacle_metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            k,
            mean_min_w_to_obstacle,
            median_min_w_to_obstacle,
            mean_min_dist,
            median_min_dist,
            num_robots_near,
            num_collisions
        ])

    if k % 50 == 0:
        print(f"iter {k}: lambda2={lambda2:.4f}, detection={detection_fraction:.2f}, meanW_target={mean_min_w_to_target:.3f}, meanW_obstacle={mean_min_w_to_obstacle:.3f}, robots_near_obs={num_robots_near}, collisions={num_collisions}")

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
import pandas as pd

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

filtered_proximity = moving_average(mean_min_w_obstacle_list, window_size=10)

time = np.arange(len(filtered_proximity))  # or your actual time vector
df = pd.DataFrame({
    "TimeStep": time,
    "FilteredProximity": filtered_proximity
})

df.to_csv("filtered_proximity.csv", index=False)
# ----------------- Post-processing plots -----------------
plt.rcParams.update({
    "font.size": 19,          # base font size
    "axes.titlesize": 19,     # title
    "axes.labelsize": 19,     # x/y labels
    "xtick.labelsize": 19,    # x ticks
    "ytick.labelsize": 19,    # y ticks
    "legend.fontsize": 16     # legend
})

plt.ioff()

# ---------------- Plot: Summary metrics (connectivity / detection / obstacle) ----------------
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(range(len(algebraic_connectivity_values)), algebraic_connectivity_values, label='λ₂')
plt.xlabel('Iteration')
plt.ylabel('Algebraic connectivity')
plt.grid(True)
plt.title('Algebraic connectivity over time')

plt.subplot(1, 3, 2)
plt.plot(range(len(detection_rates)), detection_rates, label='Detection fraction')
plt.xlabel('Iteration')
plt.ylabel('Target Detected')
plt.grid(True)
plt.title('Target Detection over time')

plt.subplot(1, 3, 3)
# obstacle-related: show mean min Wasserstein to obstacle and number of robots near obstacles (scaled y-axis)
iters = range(len(mean_min_w_obstacle_list))
plt.plot(filtered_proximity, label="Filtered Proximity")
plt.plot(iters = range(len(mean_min_w_obstacle_list)), alpha=0.3, label="Raw Proximity")  # faint raw curve
plt.legend()
plt.show()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "summary_plots_with_obstacle_metrics.pdf"))
plt.show()

# ---------------- Plot: Robot trajectories ----------------
plt.figure(figsize=(8, 6))

# HSV colormap for N robots
colors = plt.cm.hsv(np.linspace(0, 1, N))

for i in range(N):
    plt.plot(positions[i]['x'], positions[i]['y'], color=colors[i], linewidth=1.2)  # no label per-robot

# Plot targets and obstacles (legend will only show these)
if targets.size:
    plt.scatter(targets[0, :], targets[1, :], marker='*', s=180, label='Targets', color='gold')
if obstacles.size:
    plt.scatter(obstacles[0, :], obstacles[1, :], marker='o', s=150, label='Obstacles', color='red')

plt.title('Robot trajectories')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()  # only Targets and Obstacles show
plt.grid(True)
plt.savefig(os.path.join(output_dir, "trajectories.pdf"))
plt.show()

print("Simulation finished. Results saved in:", output_dir)
