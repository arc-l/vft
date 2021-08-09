import cv2
import imutils
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi
from environment import Environment
import time
import pybullet as p
from scipy import ndimage
from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
import pybullet
import utils
import pybullet_data
import time

env = Environment(gui=True)
env.reset(use_gripper=True)
p.setRealTimeSimulation(1)


# for _ in range(100000):
#     env.open_gripper()
#     time.sleep(5)
#     env.close_gripper()
#     time.sleep(5)


# img = cv2.imread("test.png")

# start_time = time.time()
# for _ in range(1000):
#     # imutils.rotate(img, 20)
#     ndimage.rotate(img, 20, reshape=False, order=0)
# end_time = time.time()
# spent = end_time - start_time
# print(f"spent {spent}")


# p0 = bc.BulletClient(connection_mode=pybullet.DIRECT)
# p1 = bc.BulletClient(connection_mode=pybullet.DIRECT)
# ur5e = p1.loadURDF("assets/ur5e/ur5e.urdf")
# robotiq_2f_85 = p0.loadURDF("assets/ur5e/gripper/robotiq_2f_85.urdf")

# ed0 = ed.UrdfEditor()
# ed0.initializeFromBulletBody(ur5e, p1._client)
# ed1 = ed.UrdfEditor()
# ed1.initializeFromBulletBody(robotiq_2f_85, p0._client)

# parentLinkIndex = 6

# jointPivotXYZInParent = [0, 0, -1]
# jointPivotRPYInParent = [0, 0, 0]

# jointPivotXYZInChild = [0, 0, 0]
# jointPivotRPYInChild = [0, -np.pi / 2, 0]

# newjoint = ed0.joinUrdf(
#     ed1,
#     parentLinkIndex,
#     jointPivotXYZInParent,
#     jointPivotRPYInParent,
#     jointPivotXYZInChild,
#     jointPivotRPYInChild,
#     p0._client,
#     p1._client,
# )
# newjoint.joint_type = p0.JOINT_FIXED

# # ed0.saveUrdf("combined.urdf")

# print(p0._client)
# print(p1._client)
# print("p0.getNumBodies()=", p0.getNumBodies())
# print("p1.getNumBodies()=", p1.getNumBodies())

# pgui = bc.BulletClient(connection_mode=pybullet.GUI)
# pgui.configureDebugVisualizer(pgui.COV_ENABLE_RENDERING, 0)

# orn = [0, 0, 0, 1]
# ed0.createMultiBody([0, 0, 0], orn, pgui._client)
# pgui.setRealTimeSimulation(1)

# pgui.configureDebugVisualizer(pgui.COV_ENABLE_RENDERING, 1)

# while pgui.isConnected():
#     pgui.getCameraImage(320, 200, renderer=pgui.ER_BULLET_HARDWARE_OPENGL)
#     time.sleep(1.0 / 240.0)


# from environment_real import EnvironmentReal

# env = EnvironmentReal()
# while True:
#     color, depth = utils.get_real_heightmap(env)
#     print(np.max(depth), np.min(depth))
#     depth[depth > 0.01] = 255
#     depth[depth <= 0.01] = 0
#     depth = depth.astype(np.uint8)
#     cv2.imshow('depth', depth)
#     cv2.imshow('color', color)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()
