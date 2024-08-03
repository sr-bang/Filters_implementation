from utils import *

filenames = [
    "Kalman_Filter\\kalman_filter_data_mocap.txt",
    "Kalman_Filter\\kalman_filter_data_low_noise.txt",
    "Kalman_Filter\\kalman_filter_data_high_noise.txt",
    "Kalman_Filter\\kalman_filter_data_velocity.txt"
]

plot_titles = [
    "Ground Truth",
    "Low Noise",
    "High Noise",
    "Velocity"
]

fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})

m = 0.027 # Mass of the quadcopter in kg
# Process noise covariance matrix (Q)
sigma_process = [0.001, 0.01, 0.01, 0.01]
# Measurement noise covariance matrix (R) 
sigma_measurement = [0.01, 0.5, 0.5, 0.5]

C_pos = np.array([1,0,0,0,0,0,
            0,1,0,0,0,0,
            0,0,1,0,0,0]).reshape(3,6)

C_vel = np.array([0,0,0,1,0,0,
            0,0,0,0,1,0,
            0,0,0,0,0,1]).reshape(3,6)

for i, filename in enumerate(filenames):
    time, u, z = load_data(filename)
    sigma = sigma_process[i]
    R = np.eye(3) * sigma_measurement[i]**2
    P = np.eye(6) * 1e-3
    
    if filename == "Kalman_Filter\\kalman_filter_data_velocity.txt":
        x_hat = np.zeros((6,1))
        x_hat[3:6,:] = z[0].reshape(3,1)       
        C = C_vel
    else:
        x_hat = np.zeros((6,1))
        x_hat[:3,:] = z[0].reshape(3,1)
        C = C_pos
        
    estimates = kalman_filter(x_hat, P, u, z, m, sigma, time, R, C)
    ax = axs[i // 2, i % 2]
    plot_trajectory(ax, estimates[:, 0], estimates[:, 1], estimates[:, 2], plot_titles[i])
plt.show()