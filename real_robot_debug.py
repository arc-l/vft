
import numpy as np
import time
from environment_real import EnvironmentReal


# User options (change me)
# --------------- Setup options ---------------
tcp_host_ip = "172.19.97.157"  # IP and port to robot arm as TCP client (UR5)
tcp_port = 30002
# ---------------------------------------------

# Initialize robot and move to home pose
env = EnvironmentReal()

# Repeatedly grasp at middle of workspace
grasp_position = np.sum(env.bounds, axis=1) / 2

grasp_position[0] = 112 * 0.002 + env.bounds[0][0]
grasp_position[1] = 112 * 0.002 + env.bounds[1][0]
grasp_position[2] = env.bounds[2][0] + 0.01

while True:
    env.push((0.0, -0.376, 0.1), (0.07, -0.336, 0.1))
    # env.grasp(grasp_position, 0 * np.pi / 8)
    time.sleep(1)

# Repeatedly move to workspace corners
# while True:
#     robot.move_to([workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]], [2.22,-2.22,0])
#     robot.move_to([workspace_limits[0][0], workspace_limits[1][1], workspace_limits[2][0]], [2.22,-2.22,0])
#     robot.move_to([workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][0]], [2.22,-2.22,0])
#     robot.move_to([workspace_limits[0][1], workspace_limits[1][0], workspace_limits[2][0]], [2.22,-2.22,0])