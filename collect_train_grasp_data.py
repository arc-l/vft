import numpy as np
import time
import cv2
import utils
import datetime
import os
import glob
import argparse
from threading import Thread
import pybullet as p
import torch

from trainer import Trainer
from constants import (
    TARGET_LOWER,
    TARGET_UPPER,
    DEPTH_MIN,
    PUSH_DISTANCE,
    GRIPPER_PUSH_RADIUS_PIXEL,
    NUM_ROTATION,
    GRASP_Q_GRASP_THRESHOLD,
    IS_REAL,
    IMAGE_PAD_WIDTH,
    IMAGE_PAD_DIFF,
)

if IS_REAL:
    from environment_real import EnvironmentReal
else:
    from environment import Environment

import multiprocessing as mp
from action_utils_mask import sample_actions, Predictor, from_maskrcnn
from train_maskrcnn import get_model_instance_segmentation


class GraspDataCollectorTrainer:
    def __init__(self, args):
        self.depth_min = DEPTH_MIN

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.base_directory = os.path.join(
            os.path.abspath("logs_grasp"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
        )
        print("Creating data logging session: %s" % (self.base_directory))
        self.color_heightmaps_directory = os.path.join(
            self.base_directory, "data", "color-heightmaps"
        )
        self.depth_heightmaps_directory = os.path.join(
            self.base_directory, "data", "depth-heightmaps"
        )
        self.mask_directory = os.path.join(self.base_directory, "data", "masks")
        self.models_directory = os.path.join(self.base_directory, "models")
        self.visualizations_directory = os.path.join(self.base_directory, "visualizations")
        self.transitions_directory = os.path.join(self.base_directory, "transitions")

        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.mask_directory):
            os.makedirs(self.mask_directory)
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory))

        self.iter = args.start_iter
        self.end_iter = args.end_iter

    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, mode):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(self.color_heightmaps_directory, "%06d.%s.color.png" % (iteration, mode)),
            color_heightmap,
        )
        depth_heightmap = np.round(depth_heightmap * 100000).astype(
            np.uint16
        )  # Save depth in 1e-5 meters
        cv2.imwrite(
            os.path.join(self.depth_heightmaps_directory, "%06d.%s.depth.png" % (iteration, mode)),
            depth_heightmap,
        )

    def write_to_log(self, log_name, log):
        np.savetxt(
            os.path.join(self.transitions_directory, "%s.log.txt" % log_name), log, delimiter=" "
        )

    def save_model(self, iteration, model, name):
        torch.save(
            {"model": model.state_dict()},
            os.path.join(self.models_directory, "snapshot-%06d.%s.pth" % (iteration, name)),
        )

    def save_backup_model(self, model, name):
        torch.save(
            {"model": model.state_dict()},
            os.path.join(self.models_directory, "snapshot-backup.%s.pth" % (name)),
        )

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(
            os.path.join(self.visualizations_directory, "%06d.%s.png" % (iteration, name)),
            affordance_vis,
        )

    def add_objects(self, env, num_obj, workspace_limits):
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
        mesh_list = glob.glob("assets/blocks/*.urdf")
        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj)
        obj_mesh_color = color_space[np.asarray(range(num_obj)) % 10, :]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        for object_idx in range(len(obj_mesh_ind)):
            curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
            drop_x = (
                (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                + workspace_limits[0][0]
                + 0.1
            )
            drop_y = (
                (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                + workspace_limits[1][0]
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

        return body_ids, True

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

    def remove_objects(self, env):
        for body_id in env.obj_ids["rigid"]:
            if p.getBasePositionAndOrientation(body_id)[0][0] < 0:
                p.removeBody(body_id)
                env.remove_object_id(body_id)

    def main(self, args, env):
        # TODO: workaround of cv2.cvtColor and pytorch dataloader, multi-thread bug
        # mp.set_start_method("spawn")

        num_obj = args.num_obj
        heightmap_resolution = env.pixel_size
        workspace_limits = env.bounds
        random_seed = args.random_seed
        force_cpu = False
        method = "reinforcement"
        push_rewards = args.push_rewards
        future_reward_discount = args.future_reward_discount
        experience_replay = args.experience_replay
        explore_rate_decay = args.explore_rate_decay
        grasp_only = args.grasp_only
        is_real = IS_REAL

        is_testing = args.is_testing
        is_grasp_explore = args.is_grasp_explore
        is_dipn = args.is_dipn
        has_target = args.has_target
        is_baseline = args.is_baseline
        max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
        test_preset_cases = args.test_preset_cases
        test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

        load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
        snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
        continue_logging = args.continue_logging  # Continue logging from previous session
        # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
        save_visualizations = args.save_visualizations

        # Set random seed
        np.random.seed(random_seed)

        # Initialize trainer
        trainer = Trainer(
            method,
            push_rewards,
            future_reward_discount,
            is_testing,
            load_snapshot,
            snapshot_file,
            force_cpu,
        )

        if is_dipn:
            # Initialize Push Prediction
            predictor = Predictor("logs_push/push_prediction_model-75.pth")
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            # Initialize Mask R-CNN
            mask_model = get_model_instance_segmentation(2)
            mask_model.load_state_dict(torch.load("logs_image/maskrcnn.pth"))
            mask_model = mask_model.to(device)
            mask_model.eval()

        # Initialize variables for heuristic bootstrapping and exploration probability
        no_change_count = [2, 2] if not is_testing else [0, 0]
        explore_prob = 0.5 if not is_testing else 0.0

        # Quick hack for nonlocal memory between threads in Python 2
        nonlocal_variables = {
            "executing_action": False,
            "primitive_action": None,
            "best_pix_ind": None,
            "push_success": False,
            "grasp_success": False,
            "primitive_position": None,
            "push_predictions": None,
            "grasp_predictions": None,
        }

        # Parallel thread to process network output and execute actions
        # -------------------------------------------------------------
        def process_actions():
            while True:
                if nonlocal_variables["executing_action"]:
                    push_predictions = nonlocal_variables["push_predictions"]
                    grasp_predictions = nonlocal_variables["grasp_predictions"]
                    if has_target:
                        grasp_predictions = trainer.focus_on_target(
                            color_heightmap,
                            valid_depth_heightmap,
                            grasp_predictions,
                            TARGET_LOWER,
                            TARGET_UPPER,
                        )

                    # Determine whether grasping or pushing should be executed based on network predictions
                    best_push_conf = np.max(push_predictions)
                    best_grasp_conf = np.max(grasp_predictions)
                    if is_dipn:
                        chosen_best_grasp_conf = best_grasp_conf
                        best_grasp_confs = np.sum(np.sort(grasp_predictions.flatten())[:])
                        print(
                            f"Before Primitive confidence scores: {best_grasp_conf} (grasp) {best_grasp_confs} (grasp sum)"
                        )
                        rotate_idx = -1
                        if best_grasp_conf < GRASP_Q_GRASP_THRESHOLD:
                            old_best_grasp_conf = best_grasp_conf
                            old_best_grasp_confs = best_grasp_confs
                            mask_objs = from_maskrcnn(mask_model, color_heightmap, device, True)
                            # if len(mask_objs) > 1 or (len(mask_objs) == 1 and best_grasp_conf < 0.5):
                            if len(mask_objs) > 1:
                                (
                                    rotated_color_image,
                                    rotated_depth_image,
                                    rotated_action,
                                    rotated_center,
                                    rotated_angle,
                                    rotated_binary_objs,
                                    before_rotated_action,
                                    rotated_mask_objs,
                                ) = sample_actions(
                                    color_heightmap, valid_depth_heightmap, mask_objs, plot=True
                                )
                                if len(rotated_color_image) > 0:
                                    (
                                        generated_color_images,
                                        generated_depth_images,
                                        validations,
                                    ) = predictor.forward(
                                        rotated_color_image,
                                        rotated_depth_image,
                                        rotated_action,
                                        rotated_center,
                                        rotated_angle,
                                        rotated_binary_objs,
                                        rotated_mask_objs,
                                        True,
                                    )
                                    for idx in range(len(generated_color_images)):
                                        if validations[idx]:
                                            with torch.no_grad():
                                                _, new_grasp_predictions = trainer.forward(
                                                    generated_color_images[idx],
                                                    generated_depth_images[idx],
                                                    is_volatile=True,
                                                    use_push=False,
                                                )
                                            if has_target:
                                                new_grasp_predictions = trainer.focus_on_target(
                                                    generated_color_images[idx],
                                                    generated_depth_images[idx],
                                                    new_grasp_predictions,
                                                    TARGET_LOWER,
                                                    TARGET_UPPER,
                                                )
                                                predicted_value = np.max(new_grasp_predictions)
                                                if chosen_best_grasp_conf < predicted_value:
                                                    rotate_idx = idx
                                                    chosen_best_grasp_conf = predicted_value
                                            else:
                                                predicted_values = np.sum(
                                                    np.sort(new_grasp_predictions.flatten())[:]
                                                )
                                                best_grasp_conf = np.max(new_grasp_predictions)
                                                if (
                                                    best_grasp_confs < predicted_values
                                                    and old_best_grasp_conf < best_grasp_conf
                                                ):
                                                    best_grasp_confs = predicted_values
                                                    rotate_idx = idx
                                                    chosen_best_grasp_conf = best_grasp_conf
                                else:
                                    print("Need to check, no action?")
                                    input("wait")
                            if has_target:
                                if rotate_idx == -1:
                                    rng = np.random.default_rng(random_seed)
                                    if np.any(validations):
                                        while True:
                                            rotate_idx = rng.integers(
                                                0, len(generated_color_images)
                                            )
                                            if validations[rotate_idx]:
                                                break
                                    else:
                                        rotate_idx = rng.integers(0, len(generated_color_images))
                                        generated_color_images[rotate_idx] = generated_color_images[
                                            rotate_idx
                                        ][
                                            IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF,
                                            IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF,
                                            :,
                                        ]
                                        generated_depth_images[rotate_idx] = generated_depth_images[
                                            rotate_idx
                                        ][
                                            IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF,
                                            IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF,
                                        ]
                                    chosen_best_grasp_conf = -10
                                if (
                                    best_grasp_confs * 0.9 < old_best_grasp_confs
                                    and chosen_best_grasp_conf > 0.5
                                ):
                                    chosen_best_grasp_conf = -1
                        # if (
                        #     rotate_idx == -1 or best_grasp_conf < old_best_grasp_conf * 1.1
                        # ) and best_grasp_conf > 0.5:
                        if rotate_idx == -1 or (
                            best_grasp_confs * 0.9 < old_best_grasp_confs
                            and chosen_best_grasp_conf > 0.5
                        ):
                            nonlocal_variables["primitive_action"] = "grasp"
                        else:
                            overlay = color_heightmap
                            added_image = cv2.addWeighted(
                                generated_color_images[rotate_idx], 0.8, overlay, 0.2, 0
                            )
                            img = cv2.cvtColor(added_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite("predict.png", img)
                            img = generated_depth_images[rotate_idx]
                            img[img <= DEPTH_MIN] = 0
                            img[img > DEPTH_MIN] = 255
                            cv2.imwrite("predictgray.png", img)
                            img = cv2.cvtColor(
                                generated_color_images[rotate_idx], cv2.COLOR_RGB2BGR
                            )
                            cv2.imwrite("predictcolor.png", img)
                            nonlocal_variables["primitive_action"] = "push"

                        print(
                            "After Primitive confidence scores: %f (grasp) %f (grasp sum)"
                            % (chosen_best_grasp_conf, best_grasp_confs)
                        )
                        trainer.is_exploit_log.append([1])
                    else:
                        print(
                            "Primitive confidence scores: %f (push), %f (grasp)"
                            % (best_push_conf, best_grasp_conf)
                        )
                        nonlocal_variables["primitive_action"] = "grasp"
                        explore_actions = False
                        if not grasp_only:
                            if best_push_conf > best_grasp_conf:
                                nonlocal_variables["primitive_action"] = "push"
                            explore_actions = np.random.uniform() < explore_prob
                            if (
                                explore_actions
                            ):  # Exploitation (do best action) vs exploration (do other action)
                                print(
                                    "Strategy: explore (exploration probability: %f)"
                                    % (explore_prob)
                                )
                                nonlocal_variables["primitive_action"] = (
                                    "push" if np.random.randint(0, 2) == 0 else "grasp"
                                )
                            else:
                                print(
                                    "Strategy: exploit (exploration probability: %f)"
                                    % (explore_prob)
                                )
                        trainer.is_exploit_log.append([0 if explore_actions else 1])
                    self.write_to_log("is-exploit", trainer.is_exploit_log)

                    use_heuristic = False

                    # Get pixel location and rotation with highest affordance prediction from
                    # heuristic algorithms (rotation, y, x)
                    if nonlocal_variables["primitive_action"] == "push":
                        if is_dipn:
                            predicted_value = best_grasp_conf
                            angle = rotated_angle[rotate_idx]
                            if angle < 0:
                                angle = 360 + angle
                            nonlocal_variables["best_pix_ind"] = (
                                int(round((angle) / (360 / NUM_ROTATION))),
                                before_rotated_action[rotate_idx][0],
                                before_rotated_action[rotate_idx][1],
                            )
                        else:
                            nonlocal_variables["best_pix_ind"] = np.unravel_index(
                                np.argmax(push_predictions), push_predictions.shape
                            )
                            predicted_value = np.max(push_predictions)
                    elif nonlocal_variables["primitive_action"] == "grasp":
                        if is_grasp_explore:
                            pow_law_exp = 1.5
                            q_lower_limit = 0.2
                            num_valid_samples = np.sum(grasp_predictions > q_lower_limit)
                            sorted_idx = np.argsort(grasp_predictions, axis=None)
                            rand_sample_idx = (
                                int(
                                    np.round(
                                        np.random.power(pow_law_exp, 1) * (num_valid_samples - 1)
                                    )
                                )
                                + sorted_idx.size
                                - num_valid_samples
                            )
                            nonlocal_variables["best_pix_ind"] = np.unravel_index(
                                sorted_idx[rand_sample_idx], grasp_predictions.shape
                            )
                            predicted_value = grasp_predictions[nonlocal_variables["best_pix_ind"]]
                            print(f"Explore grasp q value: {predicted_value} (grasp)")
                        else:
                            nonlocal_variables["best_pix_ind"] = np.unravel_index(
                                np.argmax(grasp_predictions), grasp_predictions.shape
                            )
                            predicted_value = np.max(grasp_predictions)
                    trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                    self.write_to_log("use-heuristic", trainer.use_heuristic_log)

                    # Save predicted confidence value
                    trainer.predicted_value_log.append([predicted_value])
                    self.write_to_log("predicted-value", trainer.predicted_value_log)

                    # Compute 3D position of pixel
                    print(
                        "Action: %s at (%d, %d, %d)"
                        % (
                            nonlocal_variables["primitive_action"],
                            nonlocal_variables["best_pix_ind"][0],
                            nonlocal_variables["best_pix_ind"][1],
                            nonlocal_variables["best_pix_ind"][2],
                        )
                    )
                    best_rotation_angle = np.deg2rad(
                        nonlocal_variables["best_pix_ind"][0]
                        * (360.0 / trainer.model.num_rotations)
                    )
                    best_pix_x = nonlocal_variables["best_pix_ind"][1]
                    best_pix_y = nonlocal_variables["best_pix_ind"][2]
                    primitive_position = [
                        best_pix_x * heightmap_resolution + workspace_limits[0][0],
                        best_pix_y * heightmap_resolution + workspace_limits[1][0],
                        valid_depth_heightmap[best_pix_x][best_pix_y] + workspace_limits[2][0],
                    ]

                    # If pushing, adjust start position, and make sure z value is safe and not too low
                    # or nonlocal_variables['primitive_action'] == 'place':
                    if nonlocal_variables["primitive_action"] == "push":
                        # safe_kernel_width = GRIPPER_PUSH_RADIUS_PIXEL
                        # local_region = valid_depth_heightmap[
                        #     max(best_pix_x - safe_kernel_width, 0) : min(
                        #         best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[0]
                        #     ),
                        #     max(best_pix_y - safe_kernel_width, 0) : min(
                        #         best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[1]
                        #     ),
                        # ]
                        # if local_region.size == 0:
                        #     safe_z_position = 0.01
                        # else:
                        #     safe_z_position = np.max(local_region) + 0.01
                        safe_z_position = 0.01
                        primitive_position[2] = safe_z_position

                    # Save executed primitive
                    if nonlocal_variables["primitive_action"] == "push":
                        trainer.executed_action_log.append(
                            [
                                0,
                                nonlocal_variables["best_pix_ind"][0],
                                nonlocal_variables["best_pix_ind"][1],
                                nonlocal_variables["best_pix_ind"][2],
                            ]
                        )  # 0 - push
                    elif nonlocal_variables["primitive_action"] == "grasp":
                        trainer.executed_action_log.append(
                            [
                                1,
                                nonlocal_variables["best_pix_ind"][0],
                                nonlocal_variables["best_pix_ind"][1],
                                nonlocal_variables["best_pix_ind"][2],
                            ]
                        )  # 1 - grasp
                    self.write_to_log("executed-action", trainer.executed_action_log)

                    # Visualize executed primitive, and affordances
                    if save_visualizations:
                        if not grasp_only:
                            push_pred_vis = trainer.get_prediction_vis(
                                push_predictions,
                                color_heightmap,
                                nonlocal_variables["best_pix_ind"],
                            )
                            self.save_visualizations(trainer.iteration, push_pred_vis, "push")
                            cv2.imwrite("visualization.push.png", push_pred_vis)
                        grasp_pred_vis = trainer.get_prediction_vis(
                            grasp_predictions, color_heightmap, nonlocal_variables["best_pix_ind"]
                        )
                        self.save_visualizations(trainer.iteration, grasp_pred_vis, "grasp")
                        cv2.imwrite("visualization.grasp.png", grasp_pred_vis)

                    # Initialize variables that influence reward
                    nonlocal_variables["push_success"] = False
                    nonlocal_variables["grasp_success"] = False

                    # Execute primitive
                    if nonlocal_variables["primitive_action"] == "push":
                        # primitive_position_end = [
                        #     primitive_position[0] + PUSH_DISTANCE * np.cos(-best_rotation_angle),
                        #     primitive_position[1] + PUSH_DISTANCE * np.sin(-best_rotation_angle),
                        #     primitive_position[2],
                        # ]
                        if is_dipn:
                            primitive_position_end = [
                                primitive_position[0]
                                + PUSH_DISTANCE * np.cos(-best_rotation_angle),
                                primitive_position[1]
                                + PUSH_DISTANCE * np.sin(-best_rotation_angle),
                                primitive_position[2],
                            ]
                        else:
                            primitive_position_end = [
                                primitive_position[0]
                                + PUSH_DISTANCE * np.cos(-best_rotation_angle - 180),
                                primitive_position[1]
                                + PUSH_DISTANCE * np.sin(-best_rotation_angle - 180),
                                primitive_position[2],
                            ]
                        if not is_real:
                            if env.use_gripper:
                                env.replace_to_spatula()
                            nonlocal_variables["push_success"] = env.push(
                                primitive_position, primitive_position_end
                            )
                        else:
                            nonlocal_variables["push_success"] = env.push(
                                primitive_position, primitive_position_end
                            )
                        print("Push successful: %r" % (nonlocal_variables["push_success"]))
                    elif nonlocal_variables["primitive_action"] == "grasp":
                        if not is_real:
                            if not env.use_gripper:
                                env.replace_to_gripper()
                            _, nonlocal_variables["grasp_success"] = env.grasp(
                                primitive_position, best_rotation_angle
                            )
                            self.remove_objects(env)
                        else:
                            nonlocal_variables["grasp_success"] = env.grasp(
                                primitive_position, best_rotation_angle
                            )
                        print("Grasp successful: %r" % (nonlocal_variables["grasp_success"]))

                    nonlocal_variables["primitive_position"] = (best_pix_x, best_pix_y)
                    nonlocal_variables["executing_action"] = False

                time.sleep(0.01)

        action_thread = Thread(target=process_actions)
        action_thread.daemon = True
        action_thread.start()
        exit_called = False

        # Start main training/testing loop
        if not is_real:
            env.reset(use_gripper=True)
            if test_preset_cases:
                self.add_object_push_from_file(env, test_preset_file)
            elif is_baseline:
                hard_cases = glob.glob("hard-cases/*.txt")
                self.add_object_push_from_file(env, hard_cases[trainer.iteration])
            else:
                self.add_objects(env, num_obj, workspace_limits)
        while True:
            print(
                "\n%s iteration: %d" % ("Testing" if is_testing else "Training", trainer.iteration)
            )
            iteration_time_0 = time.time()

            # Get latest RGB-D image
            if not is_real:
                color_heightmap, depth_heightmap, _ = utils.get_true_heightmap(env)
            else:
                color_heightmap, depth_heightmap = utils.get_real_heightmap(env)
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
            valid_depth_heightmap = valid_depth_heightmap.astype(np.float32)

            # Save RGB-D heightmaps
            self.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, 0)

            # Reset simulation or pause real-world training if table is empty
            stuff_count = np.zeros(valid_depth_heightmap.shape)
            stuff_count[valid_depth_heightmap > self.depth_min] = 1
            print("Stuff on the table (value: %d)" % (np.sum(stuff_count)))
            empty_threshold = 200
            if is_testing and not is_real:
                empty_threshold = 10
            if is_baseline or has_target:
                temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
                print(f"Target on the table (value: {np.sum(mask) / 255})")
                if np.sum(mask) / 255 < 50:
                    stuff_count = 0
            if np.sum(stuff_count) < empty_threshold or (
                no_change_count[0] + no_change_count[1] > 10
            ):
                no_change_count = [0, 0]
                print(
                    "Not enough objects in view (value: %d)! Repositioning objects."
                    % (np.sum(stuff_count))
                )
                if not is_real:
                    env.reset(use_gripper=True)
                    if is_baseline:
                        if np.random.uniform() < 0.2:
                            self.add_objects(env, np.random.randint(5) + 1, workspace_limits)
                        else:
                            self.add_object_push_from_file(env, hard_cases[trainer.iteration])
                    elif test_preset_cases:
                        self.add_object_push_from_file(env, test_preset_file)
                    else:
                        self.add_objects(env, num_obj, workspace_limits)
                else:
                    print(
                        "Not enough stuff on the table (value: %d)! Flipping over bin of objects..."
                        % (np.sum(stuff_count))
                    )
                    input("Please maually reset scene")
                if is_testing:  # If at end of test run, re-load original weights (before test run)
                    trainer.loss_list = []
                    trainer.optimizer.zero_grad()
                    trainer.model.load_state_dict(torch.load(snapshot_file)["model"])

                trainer.clearance_log.append([trainer.iteration])
                self.write_to_log("clearance", trainer.clearance_log)
                if is_testing and len(trainer.clearance_log) >= max_test_trials:
                    exit_called = True  # Exit after training thread (backprop and saving labels)

                if "prev_color_img" in locals():
                    # Detect changes
                    depth_diff = abs(depth_heightmap - prev_depth_heightmap)
                    depth_diff[np.isnan(depth_diff)] = 0
                    depth_diff[depth_diff > 0.3] = 0
                    depth_diff[depth_diff < 0.01] = 0
                    depth_diff[depth_diff > 0] = 1
                    change_threshold = 300
                    change_value = np.sum(depth_diff)
                    if prev_primitive_action == "push":
                        change_detected = change_value > change_threshold
                    elif prev_primitive_action == "grasp":
                        change_detected = prev_grasp_success
                    print("Change detected: %r (value: %d)" % (change_detected, change_value))

                    if change_detected:
                        if prev_primitive_action == "push":
                            no_change_count[0] = 0
                        elif prev_primitive_action == "grasp":
                            no_change_count[1] = 0
                    else:
                        if prev_primitive_action == "push":
                            no_change_count[0] += 1
                        elif prev_primitive_action == "grasp":
                            no_change_count[1] += 1

                    # Compute training labels
                    if is_baseline:
                        label_value, prev_reward_value = trainer.get_label_value_base(
                            prev_primitive_action,
                            prev_push_success,
                            prev_grasp_success,
                            change_detected,
                            prev_push_predictions,
                            prev_grasp_predictions,
                            color_heightmap,
                            valid_depth_heightmap,
                            use_push=(not grasp_only),
                            target=prev_primitive_position,
                            prev_color_img=prev_color_heightmap,
                            prev_depth_img=prev_valid_depth_heightmap,
                        )
                    else:
                        label_value, prev_reward_value = trainer.get_label_value(
                            prev_primitive_action,
                            prev_push_success,
                            prev_grasp_success,
                            change_detected,
                            prev_push_predictions,
                            prev_grasp_predictions,
                            color_heightmap,
                            valid_depth_heightmap,
                            prev_valid_depth_heightmap,
                            use_push=(not grasp_only),
                        )
                    trainer.label_value_log.append([label_value])
                    self.write_to_log("label-value", trainer.label_value_log)
                    trainer.reward_value_log.append([prev_reward_value])
                    self.write_to_log("reward-value", trainer.reward_value_log)

                    trainer.backprop(
                        prev_color_heightmap,
                        prev_valid_depth_heightmap,
                        prev_primitive_action,
                        prev_best_pix_ind,
                        label_value,
                        use_push=(not grasp_only),
                    )

                    del prev_color_img
                    nonlocal_variables["push_success"] = False
                    nonlocal_variables["grasp_success"] = False
                    nonlocal_variables["primitive_action"] = None
                    nonlocal_variables["best_pix_ind"] = None
                continue

            if not exit_called:

                # Run forward pass with network to get affordances
                push_predictions, grasp_predictions = trainer.forward(
                    color_heightmap,
                    valid_depth_heightmap,
                    is_volatile=True,
                    use_push=(not grasp_only),
                )
                nonlocal_variables["push_predictions"] = push_predictions
                nonlocal_variables["grasp_predictions"] = grasp_predictions

                # Execute best primitive action on robot in another thread
                nonlocal_variables["executing_action"] = True

            # Run training iteration in current thread (aka training thread)
            if "prev_color_img" in locals():

                # Detect changes
                depth_diff = abs(depth_heightmap - prev_depth_heightmap)
                depth_diff[np.isnan(depth_diff)] = 0
                depth_diff[depth_diff > 0.3] = 0
                depth_diff[depth_diff < 0.01] = 0
                depth_diff[depth_diff > 0] = 1
                change_threshold = 300
                change_value = np.sum(depth_diff)
                if prev_primitive_action == "push":
                    change_detected = change_value > change_threshold
                elif prev_primitive_action == "grasp":
                    change_detected = prev_grasp_success
                print("Change detected: %r (value: %d)" % (change_detected, change_value))

                if change_detected:
                    if prev_primitive_action == "push":
                        no_change_count[0] = 0
                    elif prev_primitive_action == "grasp":
                        no_change_count[1] = 0
                else:
                    if prev_primitive_action == "push":
                        no_change_count[0] += 1
                    elif prev_primitive_action == "grasp":
                        no_change_count[1] += 1

                # Compute training labels
                if is_baseline:
                    label_value, prev_reward_value = trainer.get_label_value_base(
                        prev_primitive_action,
                        prev_push_success,
                        prev_grasp_success,
                        change_detected,
                        prev_push_predictions,
                        prev_grasp_predictions,
                        color_heightmap,
                        valid_depth_heightmap,
                        use_push=(not grasp_only),
                        target=prev_primitive_position,
                        prev_color_img=prev_color_heightmap,
                        prev_depth_img=prev_valid_depth_heightmap,
                    )
                else:
                    label_value, prev_reward_value = trainer.get_label_value(
                        prev_primitive_action,
                        prev_push_success,
                        prev_grasp_success,
                        change_detected,
                        prev_push_predictions,
                        prev_grasp_predictions,
                        color_heightmap,
                        valid_depth_heightmap,
                        prev_valid_depth_heightmap,
                        use_push=(not grasp_only),
                    )
                trainer.label_value_log.append([label_value])
                self.write_to_log("label-value", trainer.label_value_log)
                trainer.reward_value_log.append([prev_reward_value])
                self.write_to_log("reward-value", trainer.reward_value_log)

                # Backpropagate
                trainer.backprop(
                    prev_color_heightmap,
                    prev_valid_depth_heightmap,
                    prev_primitive_action,
                    prev_best_pix_ind,
                    label_value,
                    use_push=(not grasp_only),
                )

                # Adjust exploration probability
                if not is_testing:
                    if is_baseline:
                        explore_prob = (
                            max(0.5 * np.power(0.9996, trainer.iteration), 0.1)
                            if explore_rate_decay
                            else 0.5
                        )
                    else:
                        explore_prob = (
                            max(0.5 * np.power(0.9998, trainer.iteration), 0.1)
                            if explore_rate_decay
                            else 0.5
                        )

                # Do sampling for experience replay
                if experience_replay and not is_testing:
                    sample_primitive_action = prev_primitive_action
                    if sample_primitive_action == "push":
                        sample_primitive_action_id = 0
                        if method == "reinforcement":
                            sample_reward_value = 0 if prev_reward_value > 0 else 0.1
                    elif sample_primitive_action == "grasp":
                        sample_primitive_action_id = 1
                        if method == "reinforcement":
                            if is_baseline:
                                sample_reward_value = 0 if prev_reward_value == 10 else 10
                            else:
                                sample_reward_value = 0 if prev_reward_value == 1 else 1

                    # Get samples of the same primitive but with different results
                    if sample_primitive_action == "push" and sample_reward_value == 0.1:
                        # sample_ind = np.argwhere(np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id)
                        sample_ind = np.argwhere(
                            np.logical_and(
                                np.asarray(trainer.reward_value_log)[: trainer.iteration - 1, 0]
                                > sample_reward_value,
                                np.asarray(trainer.executed_action_log)[: trainer.iteration - 1, 0]
                                == sample_primitive_action_id,
                            )
                        )
                    else:
                        sample_ind = np.argwhere(
                            np.logical_and(
                                np.asarray(trainer.reward_value_log)[: trainer.iteration - 1, 0]
                                == sample_reward_value,
                                np.asarray(trainer.executed_action_log)[: trainer.iteration - 1, 0]
                                == sample_primitive_action_id,
                            )
                        )
                    # don't care the reward
                    # sample_ind = np.argwhere(np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id)

                    if sample_ind.size > 0:

                        # Find sample with highest surprise value
                        if method == "reinforcement":
                            sample_surprise_values = np.abs(
                                np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]]
                                - np.asarray(trainer.label_value_log)[sample_ind[:, 0]]
                            )
                        sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
                        sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
                        pow_law_exp = 2
                        rand_sample_ind = int(
                            np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1))
                        )
                        sample_iteration = sorted_sample_ind[rand_sample_ind]
                        print(
                            "Experience replay: iteration %d (surprise value: %f)"
                            % (
                                sample_iteration,
                                sample_surprise_values[sorted_surprise_ind[rand_sample_ind]],
                            )
                        )

                        # Load sample RGB-D heightmap
                        sample_color_heightmap = cv2.imread(
                            os.path.join(
                                self.color_heightmaps_directory,
                                "%06d.0.color.png" % (sample_iteration),
                            )
                        )
                        sample_color_heightmap = cv2.cvtColor(
                            sample_color_heightmap, cv2.COLOR_BGR2RGB
                        )
                        sample_depth_heightmap = cv2.imread(
                            os.path.join(
                                self.depth_heightmaps_directory,
                                "%06d.0.depth.png" % (sample_iteration),
                            ),
                            -1,
                        )
                        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000

                        # Compute forward pass with sample
                        with torch.no_grad():
                            sample_push_predictions, sample_grasp_predictions = trainer.forward(
                                sample_color_heightmap,
                                sample_depth_heightmap,
                                is_volatile=True,
                                use_push=(not grasp_only),
                            )

                        # Load next sample RGB-D heightmap
                        next_sample_color_heightmap = cv2.imread(
                            os.path.join(
                                self.color_heightmaps_directory,
                                "%06d.0.color.png" % (sample_iteration + 1),
                            )
                        )
                        next_sample_color_heightmap = cv2.cvtColor(
                            next_sample_color_heightmap, cv2.COLOR_BGR2RGB
                        )
                        next_sample_depth_heightmap = cv2.imread(
                            os.path.join(
                                self.depth_heightmaps_directory,
                                "%06d.0.depth.png" % (sample_iteration + 1),
                            ),
                            -1,
                        )
                        next_sample_depth_heightmap = (
                            next_sample_depth_heightmap.astype(np.float32) / 100000
                        )

                        sample_reward_value = np.asarray(trainer.reward_value_log)[
                            sample_iteration, 0
                        ]
                        sample_push_success = sample_reward_value > 0
                        sample_grasp_success = sample_reward_value == 1
                        sample_change_detected = sample_push_success
                        if is_baseline:
                            sample_primitive_position = (
                                np.asarray(trainer.executed_action_log)[sample_iteration, 2:4]
                            ).astype(int)
                            (
                                new_sample_label_value,
                                new_sample_reward_value,
                            ) = trainer.get_label_value_base(
                                sample_primitive_action,
                                sample_push_success,
                                sample_grasp_success,
                                sample_change_detected,
                                sample_push_predictions,
                                sample_grasp_predictions,
                                next_sample_color_heightmap,
                                next_sample_depth_heightmap,
                                use_push=(not grasp_only),
                                target=sample_primitive_position,
                                prev_color_img=sample_color_heightmap,
                                prev_depth_img=sample_depth_heightmap,
                            )
                        else:
                            (
                                new_sample_label_value,
                                new_sample_reward_value,
                            ) = trainer.get_label_value(
                                sample_primitive_action,
                                sample_push_success,
                                sample_grasp_success,
                                sample_change_detected,
                                sample_push_predictions,
                                sample_grasp_predictions,
                                next_sample_color_heightmap,
                                next_sample_depth_heightmap,
                                sample_depth_heightmap,
                                use_push=(not grasp_only),
                            )

                        # Get labels for sample and backpropagate
                        sample_best_pix_ind = (
                            np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]
                        ).astype(int)
                        # trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])
                        trainer.backprop(
                            sample_color_heightmap,
                            sample_depth_heightmap,
                            sample_primitive_action,
                            sample_best_pix_ind,
                            new_sample_label_value,
                            use_push=(not grasp_only),
                        )

                        # Recompute prediction value and label for replay buffer
                        if sample_primitive_action == "push":
                            print(
                                "Surprise value from %f to %f"
                                % (
                                    abs(
                                        trainer.predicted_value_log[sample_iteration][0]
                                        - trainer.label_value_log[sample_iteration][0]
                                    ),
                                    abs(
                                        np.max(
                                            sample_push_predictions
                                            - trainer.label_value_log[sample_iteration][0]
                                        )
                                    ),
                                )
                            )
                            trainer.predicted_value_log[sample_iteration] = [
                                np.max(sample_push_predictions)
                            ]
                            trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                            trainer.reward_value_log[sample_iteration] = [new_sample_reward_value]
                            self.write_to_log("predicted-value", trainer.predicted_value_log)
                            self.write_to_log("reward-value", trainer.reward_value_log)
                            self.write_to_log("label-value", trainer.label_value_log)
                        elif sample_primitive_action == "grasp":
                            print(
                                "Surprise value from %f to %f"
                                % (
                                    abs(
                                        trainer.predicted_value_log[sample_iteration][0]
                                        - trainer.label_value_log[sample_iteration][0]
                                    ),
                                    abs(
                                        np.max(
                                            sample_grasp_predictions
                                            - trainer.label_value_log[sample_iteration][0]
                                        )
                                    ),
                                )
                            )
                            trainer.predicted_value_log[sample_iteration] = [
                                np.max(sample_grasp_predictions)
                            ]
                            trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                            trainer.reward_value_log[sample_iteration] = [new_sample_reward_value]
                            self.write_to_log("predicted-value", trainer.predicted_value_log)
                            self.write_to_log("reward-value", trainer.reward_value_log)
                            self.write_to_log("label-value", trainer.label_value_log)
                        print(
                            "Replay update: %f, %f, %f"
                            % (
                                trainer.predicted_value_log[sample_iteration][0],
                                trainer.label_value_log[sample_iteration][0],
                                trainer.reward_value_log[sample_iteration][0],
                            )
                        )

                    else:
                        print("Not enough prior training samples. Skipping experience replay.")

                # Save model snapshot
                if not is_testing:
                    # self.save_backup_model(trainer.model, method)
                    if trainer.iteration % 50 == 0:
                        self.save_model(trainer.iteration, trainer.model, method)
                        if trainer.use_cuda:
                            trainer.model = trainer.model.cuda()

            # Sync both action thread and training thread
            while nonlocal_variables["executing_action"]:
                time.sleep(0.01)

            if exit_called:
                break

            # Save information for next training step
            prev_color_img = color_heightmap.copy()
            prev_color_heightmap = color_heightmap.copy()
            prev_depth_heightmap = depth_heightmap.copy()
            prev_valid_depth_heightmap = valid_depth_heightmap.copy()
            prev_push_success = nonlocal_variables["push_success"]
            prev_grasp_success = nonlocal_variables["grasp_success"]
            prev_primitive_action = nonlocal_variables["primitive_action"]
            prev_primitive_position = nonlocal_variables["primitive_position"]
            if grasp_only:
                prev_push_predictions = 0
            else:
                prev_push_predictions = nonlocal_variables["push_predictions"].copy()
            prev_grasp_predictions = nonlocal_variables["grasp_predictions"].copy()
            prev_best_pix_ind = nonlocal_variables["best_pix_ind"]

            trainer.iteration += 1
            iteration_time_1 = time.time()
            print("Time elapsed: %f" % (iteration_time_1 - iteration_time_0))

            self.write_to_log("batch-loss", trainer.loss_log)
            if trainer.iteration > args.end_iter:
                exit_called = True


def post_train(args):
    import log_utils
    from collections import deque

    """
    For grasp-only training, this offline-training can be used train the network as supervised learning. But, we didn't use it.
    """
    # TODO only work for sim now
    # ------------- Algorithm options -------------
    method = "reinforcement"

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = (
        os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath("logs")
    )
    logger = log_utils.setup_logger(logging_directory)
    deque = deque(maxlen=500)

    # Initialize trainer
    trainer = Trainer(
        method,
        False,
        0,
        False,
        load_snapshot,
        snapshot_file,
        False,
    )

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(os.path.join(logging_directory, "transitions"))

    sample_ind = np.argwhere(
        np.asarray(trainer.executed_action_log)[: trainer.iteration - 1, 0] == 1
    )
    sample_primitive_action = "grasp"
    rng = np.random.default_rng()
    for i in range(100000):
        rand_sample_ind = rng.integers(low=0, high=len(sample_ind) - 1)
        sample_iteration = sample_ind[rand_sample_ind][0]

        # Load sample RGB-D heightmap
        sample_color_heightmap = cv2.imread(
            os.path.join(
                logging_directory,
                "data",
                "color-heightmaps",
                "%06d.0.color.png" % (sample_iteration),
            )
        )
        sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
        sample_depth_heightmap = cv2.imread(
            os.path.join(
                logging_directory,
                "data",
                "depth-heightmaps",
                "%06d.0.depth.png" % (sample_iteration),
            ),
            -1,
        )
        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000

        # Get labels for sample and backpropagate
        sample_best_pix_ind = (
            np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]
        ).astype(int)

        batch_loss = trainer.backprop(
            sample_color_heightmap,
            sample_depth_heightmap,
            sample_primitive_action,
            sample_best_pix_ind,
            trainer.label_value_log[sample_iteration][0],
        )

        if batch_loss != -1:
            deque.append(batch_loss)
        if i % 100 == 0:
            logger.info(f"Iteration {i}: mean {np.mean(deque)}, median {np.median(deque)}")

        if i % 500 == 0:
            torch.save(
                {"model": trainer.model.state_dict()},
                os.path.join(
                    logging_directory, "models", "snapshot-post-%06d.%s.pth" % (i, "reinforcement")
                ),
            )
            print("Saved at iteration %f" % (i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start_iter",
        dest="start_iter",
        type=int,
        action="store",
        default=0,
        help="index of start iteration",
    )
    parser.add_argument(
        "--end_iter",
        dest="end_iter",
        type=int,
        action="store",
        default=20000,
        help="index of end iteration",
    )
    parser.add_argument(
        "--num_obj",
        dest="num_obj",
        type=int,
        action="store",
        default=10,
        help="number of objects to add to simulation",
    )
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        type=int,
        action="store",
        default=123,
        help="random seed for simulation and neural net initialization",
    )
    parser.add_argument(
        "--experience_replay",
        dest="experience_replay",
        action="store_true",
        default=False,
        help="use prioritized experience replay?",
    )
    parser.add_argument(
        "--explore_rate_decay", dest="explore_rate_decay", action="store_true", default=False
    )
    parser.add_argument("--grasp_only", dest="grasp_only", action="store_true", default=False)
    parser.add_argument("--is_testing", dest="is_testing", action="store_true", default=False)
    parser.add_argument("--is_dipn", dest="is_dipn", action="store_true", default=False)
    parser.add_argument("--has_target", dest="has_target", action="store_true", default=False)
    parser.add_argument("--is_grasp_explore", action="store_true", default=False)
    parser.add_argument("--is_baseline", dest="is_baseline", action="store_true", default=False)
    parser.add_argument(
        "--max_test_trials",
        dest="max_test_trials",
        type=int,
        action="store",
        default=30,
        help="maximum number of test runs per case/scenario",
    )
    parser.add_argument(
        "--test_preset_cases", dest="test_preset_cases", action="store_true", default=False
    )
    parser.add_argument(
        "--test_preset_file", dest="test_preset_file", action="store", default="test-10-obj-01.txt"
    )
    parser.add_argument(
        "--load_snapshot",
        dest="load_snapshot",
        action="store_true",
        default=False,
        help="load pre-trained snapshot of model?",
    )
    parser.add_argument(
        "--push_rewards",
        dest="push_rewards",
        action="store_true",
        default=False,
        help="use immediate rewards (from change detection) for pushing?",
    )
    parser.add_argument(
        "--future_reward_discount",
        dest="future_reward_discount",
        type=float,
        action="store",
        default=0.5,
    )
    parser.add_argument("--snapshot_file", dest="snapshot_file", action="store")
    parser.add_argument(
        "--continue_logging",
        dest="continue_logging",
        action="store_true",
        default=False,
        help="continue logging from previous session?",
    )
    parser.add_argument("--logging_directory", dest="logging_directory", action="store")
    parser.add_argument(
        "--save_visualizations",
        dest="save_visualizations",
        action="store_true",
        default=False,
        help="save visualizations of FCN predictions?",
    )

    args = parser.parse_args()

    if not IS_REAL:
        env = Environment(gui=True)
        runner = GraspDataCollectorTrainer(args)
        runner.main(args, env)
    else:
        env = EnvironmentReal()
        runner = GraspDataCollectorTrainer(args)
        runner.main(args, env)
    # post_train(args)