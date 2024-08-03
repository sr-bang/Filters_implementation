import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file
'''
The data is in the following format:
time, u1, u2, u3, z1, z2, z3
'''
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    time, u, z = data[:, 0], data[:, 1:4], data[:, 4:7]
    return time, u, z

# State Transition Matrix
def state_transition_matrix(time): 
    '''
    The state transition matrix is given by:
    F = I + Adt
    where A is the matrix that defines the state transition dynamics
    '''
    del_t = np.mean(np.diff(time))
    A = np.array([[0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]])
    F = np.eye(6) + A * del_t
    return F

# Control Input Matrix
def control_input_matrix(time, m):
    '''
    The control input matrix is given by:
    G = B * del_t
    where B is the matrix that defines the control input dynamics
    '''
    del_t = np.mean(np.diff(time)) 
    B = np.zeros((6, 3))
    B[3:6, :] = np.eye(3) * 1 / m 
    G = B * del_t
    return G

# Predict the state and covariance matrix
def predict(u, x_hat, P, F, G, sigma):
    '''
    u: Control input
    x_hat: The mean of the state estimate
    P: The covariance matrix
    F: State transition matrix
    G: Control input matrix
    sigma: Process noise
    '''
    u = u.reshape(3, 1)
    x_hat = F @ x_hat + G @ u
    Q = sigma**2 * (G @ G.T)
    P = F @ (P @ F.T) + Q
    return x_hat, P

# Update the state and covariance matrix
def update(x_hat, C, P, R, z):
    '''
    x_hat: The mean of the state estimate
    C: Measurement matrix
    P: The covariance matrix
    R: Measurement noise
    z: Measurement
    '''
    # Kalman Gain
    K = P @ (C.T @ np.linalg.inv(C @ (P @ C.T) + R))
    z = z.reshape(3, 1)
    x_hat = x_hat + K @ (z - C @ x_hat) # Update the mean
    P = (np.eye(6) - K @ C) @ P @ (np.eye(6) - K @ C).T + K @ R @ K.T # Update the covariance matrix
    return x_hat, P

# Kalman Filter
def kalman_filter(x_hat, P, u, z, m, sigma, time, R, C):
    F = state_transition_matrix(time)
    G = control_input_matrix(time, m)

    traj = []

    for i in range(len(time)):
        x_hat, P = predict(u[i], x_hat, P, F, G, sigma)
        x_hat, P = update(x_hat, C, P, R, z[i])
        traj.append(x_hat.flatten())
    return np.array(traj)

def plot_trajectory(ax, trajectory_x, trajectory_y, trajectory_z, plot_title):
    ax.plot(trajectory_x, trajectory_y, trajectory_z)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(plot_title)

