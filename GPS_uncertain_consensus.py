import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
#import scipy.linalg as sp
import scipy.linalg as la
from scipy.linalg import sqrtm
import random
import csv
import matplotlib.pyplot as plt

np.random.seed(100)  # Set the seed to 30 for 10 robots and 95 for 5 robots
    
std_position = 0.5 # noise in the localization output
sensing_range = 0.85 #0.85 for 10 robots and 0.9 for 5 robots

# Instantiate Robotarium object
N = 10
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# How many iterations do we want (about N*0.033 seconds)
iterations = 1000

# We're working in single-integrator dynamics, and we don't want the robots
# to collide or drive off the testbed.  Thus, we're going to use barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Generated a connected graph Laplacian (for a cylce graph).
L = completeGL(N)

def wasserstein_distance(mean1, cov1, mean2, cov2):
    """
    Calculates the 2-Wasserstein distance between two multivariate normal distributions.

    Args:
    mean1 (numpy.ndarray): Mean vector of the first distribution.
    cov1 (numpy.ndarray): Covariance matrix of the first distribution.
    mean2 (numpy.ndarray): Mean vector of the second distribution.
    cov2 (numpy.ndarray): Covariance matrix of the second distribution.

    Returns:
    float: The 2-Wasserstein distance between the two distributions.
    """

    term1 = np.linalg.norm(mean1 - mean2)**2
    term2 = np.trace(cov1 + cov2 - 2 * sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1)))
    return np.sqrt(term1 + term2)
    

def hellinger_distance(mu1, cov1, mu2, cov2):
    """Calculates the Hellinger distance between two multivariate Gaussian distributions.

    Args:
        mu1 (numpy.ndarray): Mean of the first distribution.
        cov1 (numpy.ndarray): Covariance matrix of the first distribution.
        mu2 (numpy.ndarray): Mean of the second distribution.
        cov2 (numpy.ndarray): Covariance matrix of the second distribution.

    Returns:
        float: The Hellinger distance between the two distributions.
    """

    # Calculate the determinant of the covariance matrices
    det1 = la.det(cov1)
    det2 = la.det(cov2)

    # Calculate the product of the covariance matrices
    #cov_prod = la.sqrtm(cov1 @ cov2)

    # Calculate the trace of the product of the covariance matrices
    #trace_cov_prod = np.trace(cov_prod)

    # Calculate the difference between the means
    diff_mu = mu1 - mu2

    # Calculate the Hellinger distance
    exponent = - (1/8) * diff_mu.T @ la.inv((cov1 + cov2)/2) @ diff_mu

    return np.sqrt(1 - (((det1**0.25 * det2**0.25) / np.sqrt(la.det(cov1+cov2)/2)) * np.exp(exponent)))
    
    
def add_gaussian_noise(values, mean=0, std_dev=1):
    """
    Add Gaussian noise to the given values.

    Parameters:
    values (np.ndarray): The original values.
    mean (float): The mean of the Gaussian noise.
    std_dev (float): The standard deviation of the Gaussian noise.

    Returns:
    np.ndarray: The values with added Gaussian noise.
    """
    noise = np.random.normal(mean, std_dev, values.shape)
    noisy_values = values + noise

    std_array = (std_dev)*np.ones(values.shape)
    covariance_matrix = np.diag(np.sqrt(abs(noise**2)) )
    #covariance_matrix = np.diag(std_array) # std_dev is same for all robots so none of the robots gets prioritized

    #z = np.vstack((noise))
    #covariance_matrix = np.cov(z.T)
    #covariance_matrix = np.cov(noisy_values, rowvar=False)
    return noisy_values, covariance_matrix
    #example:     mu, covariance = add_gaussian_noise(means_phi, 0, 1)

def max_distance(mu1, cov1, mu2, cov2):
    """
    Add max std to the given values.

    Parameters:
    mean (float): The mean of the values.
    cov (float): The covariance matrix.

    Returns:
    np.ndarray: The distance considering the mean and cov.
    """
    mean_term = np.linalg.norm(mu1 - mu2)
    cov_term = cov1 + cov2
    #print(cov_term)

    return mean_term + np.sqrt(cov_term[0,0]) + np.sqrt(cov_term[1,1])

def min_distance(mu1, cov1, mu2, cov2):
    """
    Add max std to the given values.

    Parameters:
    mean (float): The mean of the values.
    cov (float): The covariance matrix.

    Returns:
    np.ndarray: The distance considering the mean and cov.
    """
    mean_term = np.linalg.norm(mu1 - mu2)
    cov_term = cov1 + cov2
    #print(cov_term)

    return mean_term - np.sqrt(cov_term[0,0]) - np.sqrt(cov_term[1,1])


def safety_function(x_i, x_j, threshold=0.5):
    """
    Safety function h_ij(x) for robots i and j to maintain a minimum distance.
    h_ij(x) = distance(x_i, x_j) - threshold
    """
    dist = np.linalg.norm(x_i - x_j)
    return dist - threshold

def control_barrier_function(x_i, x_j, v_i, v_j, alpha=10.0, threshold=0.5):
    """
    Apply Control Barrier Function (CBF) to enforce the safety condition.
    If the robots are too close, adjust their velocities to increase distance.
    """
    h = safety_function(x_i, x_j, threshold)
    if h < 0:
        # Compute the barrier function derivative
        dh = (x_j - x_i) / np.linalg.norm(x_j - x_i)  # direction from i to j
        # Apply CBF control law (u_i += alpha * dh)
        v_i += alpha * dh  # Adjust velocity to move robot i away from robot j
    return v_i

# Set up CSV file to record positions
with open('robot_positions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Iteration', 'Robot', 'X_Position', 'Y_Position']
    writer.writerow(header)

# Initialize a list to store positions for each robot across iterations
positions = {i: {'x': [], 'y': []} for i in range(N)}
convergence_rates = []

for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    x_si_noisy = x_si #np.zeros((2,N))
    cov = np.zeros((2,2,N))


    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    #x_si_noisy,cov = add_gaussian_noise(x_si,0,std_position)
    #print(cov)

    # For each robot... preprocess
    for i in range(N):
        # Apply Gaussian noise to the position
        x_si_noisy[:,i],cov[:,:,i] = add_gaussian_noise(x_si[:,i],0,std_position)
        if k==2: print(cov[:,:,i])

    # For each robot... controller
    for i in range(N):
        
        # Get the neighbors of robot 'i' (encoded in the graph Laplacian)
        #nj = topological_neighbors(L, i)
        nj = delta_disk_neighbors(x,i,sensing_range)
        l_nj = len(nj) + 0.01 # to avoid divide by zero error if a robot gets disconnected from all others
    	#w_ij = np.zeros(1,l_nj);        
    	#sum_wij = 0
        #compute Hellinger distance
        for j in nj:
           # Euclidean distance norm-2
           if 100 <= k <= 200:
               x_si_noisy[:, i] = [0, 0]
               x_si_noisy[:, j] = [0, 0]
           d_ij = np.linalg.norm(x_si_noisy[:, j] - x_si_noisy[:, i])
           
           # Hellinger distance
           #d_ij = hellinger_distance(x_si_noisy[:,i],cov[:,:,i],x_si_noisy[:, j],cov[:,:,j])
              
           # Wasserstein distance
           #d_ij = wasserstein_distance(x_si_noisy[:,i],cov[:,:,i],x_si_noisy[:, j],cov[:,:,j])
           
           # Max distance
           #d_ij = max_distance(x_si_noisy[:,i],cov[:,:,i],x_si_noisy[:, j],cov[:,:,j])
           
           #print(d_ij)
           #calculate weights
           w_ij = (2*sensing_range**2) /((sensing_range**2 - d_ij**2)**2) #Calculating weights for each neighbor
	   #sum_wij += w_ij
           w_ij = (w_ij/l_nj) 
           # Compute the consensus algorithm
           #si_velocities[:, i] += 0.2 * (np.sum(x_si_noisy[:, j] - x_si_noisy[:, i, None], 1))
           si_velocities[:,i] += 0.4 * w_ij * (x_si_noisy[:, j] - x_si_noisy[:, i])


    # Use the barrier certificate to avoid collisions
    si_velocities = si_barrier_cert(si_velocities, x_si_noisy)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation

    print(x_si_noisy, "x_si_noisy")
    
    fixed_centroid =  np.array([-0.25, 0])
     # Compute convergence rate using the fixed centroid
    convergence_rate = np.mean(np.linalg.norm(x_si_noisy - fixed_centroid[:, None], axis=0)) - 0.5
    convergence_rates.append(convergence_rate)

    

    r.step()

       # Record positions into CSV file and store positions for plotting
    with open('robot_positions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(N):
            writer.writerow([k, i, x[0, i], x[1, i]])
            positions[i]['x'].append(x[0, i])  # Store x position
            positions[i]['y'].append(x[1, i])  # Store y position

print(positions[i]['x'], "positions[i]['x']")
# Debug: Ensure all robots have full trajectories
for i in range(N):
    print(f"Robot {i} - x length: {len(positions[i]['x'])}, y length: {len(positions[i]['y'])}")

# After the simulation finishes, plot the trajectories
plt.figure(figsize=(12, 8))

# Subplot 1: Plot x-components of positions over time
plt.subplot(2, 1, 1)
for i in range(N):
    # Plot the trajectory of robot i's x-position over time
    plt.plot(range(iterations), positions[i]['x'], label=f'Robot {i}')
plt.title('Robot X-Positions Over Time')
plt.xlabel('Time Step')
plt.ylabel('X Position')
plt.legend()
plt.grid(True)

# Subplot 2: Plot y-components of positions over time
plt.subplot(2, 1, 2)
for i in range(N):
    # Plot the trajectory of robot i's y-position over time
    plt.plot(range(iterations), positions[i]['y'], label=f'Robot {i}')
plt.title('Robot Y-Positions Over Time')
plt.xlabel('Time Step')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

# Smooth the convergence_rates using a moving average filter
window_size = 20  # Adjust the window size as needed
window = np.ones(window_size) / window_size
convergence_rates_smoothed = np.convolve(convergence_rates, window, mode='valid')

# Record the smoothed convergence rates and other metrics in a CSV file
with open('performance_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration",  "Convergence Rate (Smoothed)"])

    for i in range(len(convergence_rates_smoothed)):
        # Adjust the index for smoothed data to align with the correct iteration
        iteration = i + window_size // 2
        writer.writerow([iteration, convergence_rates_smoothed[i]])

plt.figure()
plt.plot(np.arange(len(convergence_rates_smoothed)) + window_size // 2, convergence_rates_smoothed, label="Convergence Rate (Smoothed)")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Metric Value")
plt.title("Performance Metrics")
plt.show()


input("Press Enter to close...")  # Pause until user input

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()


