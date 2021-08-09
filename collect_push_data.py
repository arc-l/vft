import time
import datetime
import os
import glob
import pybullet as p
import numpy as np
import cv2
import utils

from environment import Environment
from constants import (
    DEPTH_MIN,
    PUSH_DISTANCE,
    IMAGE_SIZE,
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
)


class PushDataCollector:
    def __init__(self, start_iter=0, end_iter=2000, base_directory=None, seed=0):
        # Objects have heights of 0.05 meters, so center should be less than 0.035
        self.height_upper = 0.035
        self.depth_min = DEPTH_MIN

        self.rng = np.random.default_rng(seed)

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        if base_directory is None:
            self.base_directory = os.path.join(
                os.path.abspath("logs_push"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
            )
        else:
            self.base_directory = base_directory
        print("Creating data logging session: %s" % (self.base_directory))
        self.prev_color_heightmaps_directory = os.path.join(
            self.base_directory, "data", "prev-color-heightmaps"
        )
        self.prev_depth_heightmaps_directory = os.path.join(
            self.base_directory, "data", "prev-depth-heightmaps"
        )
        self.prev_pose_directory = os.path.join(self.base_directory, "data", "prev-poses")
        self.next_color_heightmaps_directory = os.path.join(
            self.base_directory, "data", "next-color-heightmaps"
        )
        self.next_depth_heightmaps_directory = os.path.join(
            self.base_directory, "data", "next-depth-heightmaps"
        )
        self.next_pose_directory = os.path.join(self.base_directory, "data", "next-poses")
        self.action_directory = os.path.join(self.base_directory, "data", "actions")
        self.mask_directory = os.path.join(self.base_directory, "data", "masks")

        if not os.path.exists(self.prev_color_heightmaps_directory):
            os.makedirs(self.prev_color_heightmaps_directory)
        if not os.path.exists(self.prev_depth_heightmaps_directory):
            os.makedirs(self.prev_depth_heightmaps_directory)
        if not os.path.exists(self.prev_pose_directory):
            os.makedirs(self.prev_pose_directory)
        if not os.path.exists(self.next_color_heightmaps_directory):
            os.makedirs(self.next_color_heightmaps_directory)
        if not os.path.exists(self.next_depth_heightmaps_directory):
            os.makedirs(self.next_depth_heightmaps_directory)
        if not os.path.exists(self.next_pose_directory):
            os.makedirs(self.next_pose_directory)
        if not os.path.exists(self.action_directory):
            os.makedirs(self.action_directory)
        if not os.path.exists(self.mask_directory):
            os.makedirs(self.mask_directory)

        self.iter = start_iter
        self.end_iter = end_iter

    def reset_np_random(self, seed):
        self.rng = np.random.default_rng(seed)

    def save_heightmaps(
        self,
        iteration,
        prev_color_heightmap,
        prev_depth_heightmap,
        next_color_heightmap,
        next_depth_heightmap,
    ):
        color_heightmap = cv2.cvtColor(prev_color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.prev_color_heightmaps_directory, "%07d.color.png" % (iteration)),
            color_heightmap,
        )
        depth_heightmap = np.round(prev_depth_heightmap * 100000).astype(
            np.uint16
        )  # Save depth in 1e-5 meters
        cv2.imwrite(
            os.path.join(self.prev_depth_heightmaps_directory, "%07d.depth.png" % (iteration)),
            depth_heightmap,
        )

        color_heightmap = cv2.cvtColor(next_color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.next_color_heightmaps_directory, "%07d.color.png" % (iteration)),
            color_heightmap,
        )
        depth_heightmap = np.round(next_depth_heightmap * 100000).astype(
            np.uint16
        )  # Save depth in 1e-5 meters
        cv2.imwrite(
            os.path.join(self.next_depth_heightmaps_directory, "%07d.depth.png" % (iteration)),
            depth_heightmap,
        )

    def save_masks(self, iteration, mask):
        cv2.imwrite(os.path.join(self.mask_directory, "%07d.mask.png" % (iteration)), mask)

    def save_action(self, iteration, pose):
        np.savetxt(
            os.path.join(self.action_directory, "%07d.action.txt" % (iteration)), pose, fmt="%s"
        )

    def save_pose(self, iteration, pose0, pose1):
        np.savetxt(
            os.path.join(self.prev_pose_directory, "%07d.pose.txt" % (iteration)), pose0, fmt="%s"
        )
        np.savetxt(
            os.path.join(self.next_pose_directory, "%07d.pose.txt" % (iteration)), pose1, fmt="%s"
        )

    def add_object_push_from_file(self, env, file_name):
        body_ids = []
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            obj_files = []
            obj_mesh_colors = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                obj_file = os.path.join("assets", "blocks", file_content_curr_object[0])
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[7]),
                        float(file_content_curr_object[8]),
                        float(file_content_curr_object[9]),
                    ]
                )
                obj_mesh_colors.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            object_color = [
                obj_mesh_colors[object_idx][0],
                obj_mesh_colors[object_idx][1],
                obj_mesh_colors[object_idx][2],
                1,
            ]
            body_id = p.loadURDF(
                curr_mesh_file, object_position, p.getQuaternionFromEuler(object_orientation)
            )
            p.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            env.add_object_id(body_id)
            success &= env.wait_static()
        success &= env.wait_static()

        # give time to stop
        for _ in range(5):
            p.stepSimulation(env.client_id)

        return body_ids, success

    def add_object_push(self, env):
        """Randomly dropped objects to the workspace"""
        color_space = (
            np.asarray(
                [
                    [78.0, 121.0, 167.0],  # blue
                    [89.0, 161.0, 79.0],  # green
                    [156, 117, 95],  # brown
                    [242, 142, 43],  # orange
                    [237.0, 201.0, 72.0],  # yellow
                    [186, 176, 172],  # gray
                    [255.0, 87.0, 89.0],  # red
                    [176, 122, 161],  # purple
                    [118, 183, 178],  # cyan
                    [255, 157, 167],  # pink
                ]
            )
            / 255.0
        )
        drop_height = 0.15
        obj_num = self.rng.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.01, 0.04, 0.05, 0.15, 0.20, 0.20, 0.15, 0.1, 0.1]
        )
        mesh_list = glob.glob("assets/blocks/*.urdf")
        obj_mesh_ind = self.rng.choice(mesh_list, obj_num)
        obj_mesh_color = color_space[np.asarray(range(obj_num)), :]
        obj_mesh_color_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        object_positions = []
        object_orientations = []
        success = True
        for object_idx in range(len(obj_mesh_ind)):
            curr_mesh_file = obj_mesh_ind[object_idx]
            drop_x = 0.45 + self.rng.random() * 0.1
            drop_y = -0.05 + self.rng.random() * 0.1
            object_position = [drop_x, drop_y, drop_height]
            object_orientation = [0, 0, 2 * np.pi * self.rng.random()]
            adjust_angle = 2 * np.pi * self.rng.random()
            object_color = [
                obj_mesh_color[obj_mesh_color_ind[object_idx]][0],
                obj_mesh_color[obj_mesh_color_ind[object_idx]][1],
                obj_mesh_color[obj_mesh_color_ind[object_idx]][2],
                1,
            ]
            body_id = p.loadURDF(
                curr_mesh_file, object_position, p.getQuaternionFromEuler(object_orientation)
            )
            p.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            env.add_object_id(body_id)
            success &= env.wait_static()

            count = 0
            while success:
                success &= env.wait_static()
                object_position, _ = p.getBasePositionAndOrientation(body_id)
                if count > 20:
                    break
                # if overlap
                if object_position[2] > self.height_upper:
                    drop_x = np.cos(adjust_angle) * 0.01 + drop_x  # 1 cm
                    drop_y = np.sin(adjust_angle) * 0.01 + drop_y
                    object_position = [drop_x, drop_y, drop_height]
                    p.resetBasePositionAndOrientation(
                        body_id, object_position, p.getQuaternionFromEuler(object_orientation)
                    )
                else:
                    break
                count += 1
            if count > 20:
                object_position = [drop_x, drop_y, self.height_upper + 0.01]
                p.resetBasePositionAndOrientation(
                    body_id, object_position, p.getQuaternionFromEuler(object_orientation)
                )
            object_position, _ = p.getBasePositionAndOrientation(body_id)
            object_positions.append(object_position)
            object_orientations.append(object_orientation)

        for idx in range(len(body_ids)):
            p.resetBasePositionAndOrientation(
                body_ids[idx],
                object_positions[idx],
                p.getQuaternionFromEuler(object_orientations[idx]),
            )
            success &= env.wait_static()

        # give time to stop
        for _ in range(5):
            p.stepSimulation(env.client_id)

        return body_ids, success

    def is_valid(self, body_ids, env):
        """Decide randomly dropped objects in the valid state."""
        for body_id in body_ids:
            # Check height
            object_position, object_orientation = p.getBasePositionAndOrientation(body_id)
            if object_position[2] > self.height_upper:
                print(f"Height is wrong. Skip! {object_position[2]} > {self.height_upper}")
                return False
            # Check range
            if (
                object_position[0] < env.bounds[0][0] + PUSH_DISTANCE / 2
                or object_position[0] > env.bounds[0][1] - PUSH_DISTANCE / 2
                or object_position[1] < env.bounds[1][0] + PUSH_DISTANCE / 2
                or object_position[1] > env.bounds[1][1] - PUSH_DISTANCE / 2
            ):
                print(f"Out of bounds. Skip! {object_position[0]}, {object_position[1]}")
                return False
            # Check orientation
            object_orientation = p.getEulerFromQuaternion(object_orientation)
            if abs(object_orientation[0]) > 1e-2 or abs(object_orientation[1]) > 1e-2:
                print(f"Wrong orientation. Skip! {object_orientation}")
                return False
        return True

    def get_push_action(self, depth):
        """Find target and push, the robot makes a push from left to right."""
        depth_heightmap = np.copy(depth)
        depth_heightmap[depth_heightmap <= self.depth_min] = 0
        depth_heightmap[depth_heightmap > self.depth_min] = 1

        y_indices = np.argwhere(depth_heightmap == 1)[:, 1]  # Find the y range
        if len(y_indices) == 0:
            print("find Skip")
            return None
        y_list_unique, y_list_count = np.unique(y_indices, return_counts=True)
        y_list_dist = y_list_count / y_list_count.sum()
        y = self.rng.choice(y_list_unique, p=y_list_dist)
        x_indices = np.argwhere(depth_heightmap[:, y] == 1)[:, 0]  # Find the x range
        x_indices_left = np.argwhere(
            depth_heightmap[:, max(0, y - GRIPPER_PUSH_RADIUS_PIXEL)] == 1
        )[
            :, 0
        ]  # Find the x range
        x_indices_right = np.argwhere(
            depth_heightmap[:, min(y + GRIPPER_PUSH_RADIUS_PIXEL, IMAGE_SIZE - 1)] == 1
        )[
            :, 0
        ]  # Find the x range
        if len(x_indices) == 0:
            print("Skip 1")
            return None
        x = x_indices.min()
        if len(x_indices_left) != 0:
            x = min(x, x_indices_left.min())
        if len(x_indices_right) != 0:
            x = min(x, x_indices_right.min())
        x = x - GRIPPER_PUSH_RADIUS_SAFE_PIXEL
        if x <= 0:
            print("Skip 2")
            return None

        safe_z_position = 0.01
        return [
            x * env.pixel_size + env.bounds[0][0],
            y * env.pixel_size + env.bounds[1][0],
            safe_z_position,
        ]

    def get_poses(self, body_ids):
        poses = []
        for body_id in body_ids:
            pos, rot = p.getBasePositionAndOrientation(body_id)
            rot = p.getEulerFromQuaternion(rot)
            poses.append(pos[0])
            poses.append(pos[1])
            poses.append(rot[0])
            poses.append(rot[1])
            poses.append(rot[2])
        return poses


if __name__ == "__main__":

    is_test = False

    env = Environment(gui=False)
    if is_test:
        collector = PushDataCollector(start_iter=0, end_iter=2000)
        cases = sorted(glob.glob("hard-cases-test/*.txt"))
    else:
        collector = PushDataCollector(start_iter=0, end_iter=200000)
        cases = sorted(glob.glob("hard-cases/*.txt"))
    cases_idx = 0
    num_cases = len(cases)

    if is_test:
        seed = 200000
    else:
        seed = 0

    # multi_thread_start = 160
    # multi_thread_end = multi_thread_start + 40
    # collector.iter += multi_thread_start
    # seed += multi_thread_start
    # cases_idx += multi_thread_start

    while collector.iter < collector.end_iter:
        # if collector.iter > multi_thread_end:
        #     break

        print(f"-----Collecting: {collector.iter + 1}/{collector.end_iter}-----")
        collector.reset_np_random(seed)
        env.reset(use_gripper=False)
        # add objects, some from hard cases and some from random cases
        if collector.iter > collector.end_iter // 5:
            body_ids, success = collector.add_object_push_from_file(env, cases[cases_idx])
            cases_idx += 1
            if cases_idx == num_cases:
                cases_idx = 0
        else:
            body_ids, success = collector.add_object_push(env)
        if success and collector.is_valid(body_ids, env):
            # record info0
            color0, depth0, segm0 = utils.get_true_heightmap(env)
            poses0 = collector.get_poses(body_ids)
            # push
            action = collector.get_push_action(depth0)
            if action is not None:
                action_end = [action[0] + PUSH_DISTANCE, action[1], action[2]]
                success = env.push(action, action_end)
                success &= env.wait_static()
                success &= collector.is_valid(body_ids, env)
                if success:
                    # record info1
                    color1, depth1, segm1 = utils.get_true_heightmap(env)
                    poses1 = collector.get_poses(body_ids)
                    # save data
                    collector.save_heightmaps(collector.iter, color0, depth0, color1, depth1)
                    collector.save_action(collector.iter, [action])
                    collector.save_pose(collector.iter, [poses0], [poses1])
                    # >>>>> save masks
                    # segm_ids = np.unique(segm1)
                    # for sid in segm_ids:
                    #     if sid not in body_ids:
                    #         segm1[segm1 == sid] = 0
                    # bidxs = []
                    # for bid in body_ids:
                    #     bidxs.append(segm1 == bid)
                    # for idx, bidx in enumerate(bidxs):
                    #     segm1[bidx] = idx + 1
                    # collector.save_masks(collector.iter, segm1)
                    # <<<<<
                    collector.iter += 1
        seed += 1
