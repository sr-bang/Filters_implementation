import scipy.io
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# calculate world coordinates from AprilTag IDs

# in meters
def calculate_world_co(tag_ids):
    grid_size = [12,9]
    tag_size = 0.152 # also vertical spacing
    spacing_y= [0.152, 0.152, 0.178, 0.152, 0.152, 0.178, 0.152, 0.152] # horizontal spacing
    world_coordinates = []

    # position of the tags in the grid
    for tag_id in tag_ids:
        row = tag_id % grid_size[0]
        col = tag_id // grid_size[0]

        # calculate x, y, z coordinates
        x_offset = row * tag_size
        y_offset = sum(spacing_y[:col+1])
        z = 0.0

        # Corner coordinates in the local tag frame
        x1, y1 = row * tag_size, col * tag_size
        x2, y2 = row * tag_size + tag_size, col * tag_size
        x3, y3 = row * tag_size + tag_size, col * tag_size + tag_size
        x4, y4 = row * tag_size, col * tag_size + tag_size

        img_coor = np.array(
            [
                [x1, y1, 0.0],
                [x2, y2, 0.0],
                [x3, y3, 0.0],
                [x4, y4, 0.0],
            ]
        )

        # Transforming local coordinates to world coordinates
        world_coor = img_coor + np.array([x_offset, y_offset, z])
        world_coordinates.extend(map(tuple, world_coor))

    return world_coordinates


# Task 1: Pose Estimation

def estimate_pose(data):
    p1 = data['p4']
    p2 = data['p1']
    p3 = data['p2']
    p4 = data['p3']
    img_pts = np.array([p1, p2, p3, p4])
    tag_ids =data['id']

    # If the tag id is int, convert it into an array
    if isinstance(tag_ids, int):
        tag_ids = np.array([tag_ids])
    
    # Reshape the image data points into a 2x1 array
    points = [pt.reshape(2, 1) for pt in points]
    points = np.array(points)

    ts = []
    # if no tag found, return nan
    if(len(tag_ids)==0):
        position, orientation, timestamp = np.nan, np.nan, np.nan
        return position, orientation ,timestamp
    
    img_pts_2d = []
    for pt in points:
        img_pts_2d.append((pt[0, 0], pt[1, 0]))
    img_pts_2d = np.array(img_pts_2d)
    
    # Calculate the world coordinates of the tags
    world_pts = calculate_world_co(tag_ids)
    world_pts = np.array(world_pts)

    camera_matrix = np.array([[314.1779 , 0 , 199.4848],
                              [0 , 314.2218 , 113.7838],
                              [0, 0, 1]])

    dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])

    # Estimate the pose
    success, rvec, tvec = cv2.solvePnP(world_pts, img_pts_2d, camera_matrix, dist_coeffs)
    if not success:
        raise RuntimeError("PnP solver failed to estimate the pose")
    
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    T_cam_world = np.hstack((R, tvec))
    T_cam_world = np.vstack((T_cam_world, [0, 0, 0, 1]))

    yaw = np.pi/4
    rotation_yaw = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1] 
                ])
    
    rotation_roll = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
    
    rot = rotation_roll @ rotation_yaw 

    T_cam_imu = np.hstack([rot,np.array([-0.04,0,-0.03]).reshape(3,1)])

    T_cam_imu = np.vstack((T_cam_imu, [0, 0, 0, 1]))

    T_world_imu = np.linalg.inv(T_cam_world) @ T_cam_imu

    # Converting rotation matrix to euler angles 
    position = T_world_imu[:3, 3]

    r = Rotation.from_matrix(T_world_imu[:3, :3])
    roll, pitch, yaw = r.as_euler('xyz')

    orientation = np.array([roll, pitch, yaw])

    return position, orientation, timestamp

# Task 2: Visualization
def load_data(filename):
    return scipy.io.loadmat(filename, simplify_cells=True)


def visualize_trajectory(filename):
    mat_data = load_data(filename)

    # Get vicon data
    vicon_data = mat_data['vicon']
    vicon_data = np.array(vicon_data).T
    time = mat_data['time']

    ground_truth_positions = vicon_data[:, :3]
    estimated_pos = []
    estimated_ori = []
    pose_estimation_timestamp = []
    
    count = 0
    for i in range(len(time)):
        data = mat_data['data'][i]
        position, orientation, timestamp = estimate_pose(data)
        if not np.isnan(position).any() and not np.isnan(orientation).any() and not np.isnan(timestamp).any():
            estimated_pos.append(position)
            estimated_ori.append(orientation)
            pose_estimation_timestamp.append(timestamp)
            count += 1
        
    estimated_pos = np.array(estimated_pos)
    estimated_ori = np.array(estimated_ori)

    # # Plot 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(estimated_pos[:, 0], estimated_pos[:, 1], estimated_pos[:, 2], label='Estimated', color='red')
    ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Ground Truth vs Estimated")
    fig.savefig(f"Outputs\\{filename}_trajectory.png", dpi=300)  # Save the 3D plot
    plt.tight_layout()

    # Plot euler angles
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
    axs[0].plot(pose_estimation_timestamp, estimated_ori[:, 0], label='Estimated', color='red')
    axs[0].plot(time, vicon_data[:, 3], label='Ground Truth', color='blue')
    axs[0].set_ylabel('Phi') # roll
    axs[0].legend()

    axs[1].plot(pose_estimation_timestamp, estimated_ori[:, 1], label='Estimated', color='red')
    axs[1].plot(time, vicon_data[:, 4], label='Ground Truth', color='blue')
    axs[1].set_ylabel('Theta') # pitch
    axs[1].legend()

    axs[2].plot(pose_estimation_timestamp, estimated_ori[:, 2], label='Estimated', color='red')
    axs[2].plot(time, vicon_data[:, 5], label='Ground Truth', color='blue')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Psi)') #yaw
    axs[2].legend()

    plt.tight_layout()
    fig.savefig(f"Outputs\\{filename}_rpy.png")  # Save the ori plot


# filenames = [
#         'studentdata0.mat',
#         'studentdata1.mat',
#         'studentdata2.mat',
#         'studentdata3.mat',
#         'studentdata4.mat',
#         'studentdata5.mat',
#         'studentdata6.mat',
#         'studentdata7.mat'
# ]

# # Task 1 & 2:
# for filename in filenames:
    # visualize_trajectory(filename)