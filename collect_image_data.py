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
    COLOR_SPACE,
    WORKSPACE_LIMITS,
)


class ImageDataCollector:
    def __init__(self, start_iter=0, end_iter=2000, base_directory=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.mesh_list = glob.glob("assets/blocks/*.urdf")

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        if base_directory is None:
            self.base_directory = os.path.join(
                os.path.abspath("logs_image"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
            )
        else:
            self.base_directory = base_directory
        print("Creating data logging session: %s" % (self.base_directory))
        self.color_heightmaps_directory = os.path.join(
            self.base_directory, "data", "color-heightmaps"
        )
        self.depth_heightmaps_directory = os.path.join(
            self.base_directory, "data", "depth-heightmaps"
        )
        self.mask_directory = os.path.join(self.base_directory, "data", "masks")

        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.mask_directory):
            os.makedirs(self.mask_directory)

        self.iter = start_iter
        self.end_iter = end_iter

    def reset_np_random(self, seed):
        self.rng = np.random.default_rng(seed)

    def save_heightmaps(
        self,
        iteration,
        color_heightmap,
        depth_heightmap,
    ):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.color_heightmaps_directory, "%07d.color.png" % (iteration)),
            color_heightmap,
        )
        depth_heightmap = np.round(depth_heightmap * 100000).astype(
            np.uint16
        )  # Save depth in 1e-5 meters
        cv2.imwrite(
            os.path.join(self.depth_heightmaps_directory, "%07d.depth.png" % (iteration)),
            depth_heightmap,
        )

    def save_masks(self, iteration, mask):
        cv2.imwrite(os.path.join(self.mask_directory, "%07d.mask.png" % (iteration)), mask)

    def add_objects(self, env, num_obj):
        """Randomly dropped objects to the workspace"""
        obj_mesh_ind = self.rng.integers(0, len(self.mesh_list), size=num_obj)
        obj_mesh_color = COLOR_SPACE[np.asarray(range(num_obj)) % 10, :]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        for object_idx in range(len(obj_mesh_ind)):
            curr_mesh_file = self.mesh_list[obj_mesh_ind[object_idx]]
            drop_x = (
                (WORKSPACE_LIMITS[0][1] - WORKSPACE_LIMITS[0][0] - 0.2) * np.random.random_sample()
                + WORKSPACE_LIMITS[0][0]
                + 0.1
            )
            drop_y = (
                (WORKSPACE_LIMITS[1][1] - WORKSPACE_LIMITS[1][0] - 0.2) * np.random.random_sample()
                + WORKSPACE_LIMITS[1][0]
                + 0.1
            )
            object_position = [drop_x, drop_y, 0.2]
            object_orientation = [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
            ]
            object_color = [
                obj_mesh_color[object_idx][0],
                obj_mesh_color[object_idx][1],
                obj_mesh_color[object_idx][2],
                1,
            ]
            body_id = p.loadURDF(
                curr_mesh_file, object_position, p.getQuaternionFromEuler(object_orientation)
            )
            p.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            env.add_object_id(body_id)
            env.wait_static()

        return body_ids

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

        color_idx = random.randint(num_obj)
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
                obj_mesh_colors[color_idx][0],
                obj_mesh_colors[color_idx][1],
                obj_mesh_colors[color_idx][2],
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


if __name__ == "__main__":

    env = Environment(gui=False)
    collector = ImageDataCollector(start_iter=0, end_iter=8000)
    seed = 0

    # multi_thread_start = 4000
    # multi_thread_end = multi_thread_start + 4000
    # collector.iter += multi_thread_start
    # seed += multi_thread_start

    while collector.iter < collector.end_iter:
        # if collector.iter > multi_thread_end:
        #     break

        print(f"-----Collecting: {collector.iter + 1}/{collector.end_iter}-----")
        collector.reset_np_random(seed)
        env.reset(use_gripper=False)
        # add objects
        num_objs = collector.rng.integers(1, 11, size=1)[0]
        body_ids = collector.add_objects(env, num_objs)
        success = env.wait_static()
        if success:
            # record info0
            color0, depth0, segm0 = utils.get_true_heightmap(env)
            # save data
            collector.save_heightmaps(collector.iter, color0, depth0)
            segm_ids = np.unique(segm0)
            for sid in segm_ids:
                if sid not in body_ids:
                    segm0[segm0 == sid] = 0
            bidxs = []
            for bid in body_ids:
                bidxs.append(segm0 == bid)
            for idx, bidx in enumerate(bidxs):
                segm0[bidx] = idx + 1
            collector.save_masks(collector.iter, segm0)
            collector.iter += 1
        seed += 1
