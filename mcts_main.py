"""Test"""

import glob
import math
import gc
import os
import time
import datetime
import pybullet as p
import cv2
import numpy as np
from graphviz import Digraph
import argparse
import random
import torch

from push_predictor import PushPredictor
from mcts.search import MonteCarloTreeSearch
from mcts.nodes import PushSearchNode
from mcts.push import PushState, is_consecutive
import utils
from constants import (
    PIXEL_SIZE,
    WORKSPACE_LIMITS,
    TARGET_LOWER,
    TARGET_UPPER,
    NUM_ROTATION,
    GRASP_Q_PUSH_THRESHOLD,
    GRASP_Q_GRASP_THRESHOLD,
    IS_REAL,
    MCTS_MAX_LEVEL,
    MCTS_ROLLOUTS,
    MCTS_DISCOUNT,
    BG_THRESHOLD,
    MCTS_TOP,
)

if IS_REAL:
    from environment_real import EnvironmentReal
else:
    from environment import Environment


class SeachCollector:
    def __init__(self):
        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.base_directory = os.path.join(
            os.path.abspath("logs_grasp"), "mcts-" + timestamp_value.strftime("%Y-%m-%d-%H-%M-%S")
        )
        print("Creating data logging session: %s" % (self.base_directory))
        self.color_heightmaps_directory = os.path.join(
            self.base_directory, "data", "color-heightmaps"
        )
        self.depth_heightmaps_directory = os.path.join(
            self.base_directory, "data", "depth-heightmaps"
        )
        self.mask_directory = os.path.join(self.base_directory, "data", "masks")
        self.prediction_directory = os.path.join(self.base_directory, "data", "predictions")
        self.visualizations_directory = os.path.join(self.base_directory, "visualizations")
        self.transitions_directory = os.path.join(self.base_directory, "transitions")
        self.executed_action_log = []
        self.label_value_log = []
        self.consecutive_log = []

        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.mask_directory):
            os.makedirs(self.mask_directory)
        if not os.path.exists(self.prediction_directory):
            os.makedirs(self.prediction_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory))

    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, mode=0):
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

    def save_predictions(self, iteration, pred, name="push"):
        cv2.imwrite(
            os.path.join(self.prediction_directory, "%06d.png" % (iteration)),
            pred,
        )

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(
            os.path.join(self.visualizations_directory, "%06d.%s.png" % (iteration, name)),
            affordance_vis,
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


def parse_args():
    parser = argparse.ArgumentParser(description="MCTS DIPN")

    parser.add_argument("--test_case", action="store", help="File for testing")

    parser.add_argument(
        "--max_test_trials",
        action="store",
        type=int,
        default=5,
        help="maximum number of test runs per case/scenario",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # set seed
    random.seed(12345)
    torch.manual_seed(12345)

    iteration = 0
    args = parse_args()
    case = args.test_case
    repeat_num = args.max_test_trials
    collector = SeachCollector()
    predictor = PushPredictor()
    if IS_REAL:
        env = EnvironmentReal()
    else:
        env = Environment(gui=True)

    for repeat_idx in range(repeat_num):
        if not IS_REAL:
            success = False
            while not success:
                env.reset()
                success = collector.add_object_push_from_file(env, case)
                print(f"Reset environment at iteration {iteration} of repeat times {repeat_idx}")
        else:
            print(f"Reset environment at iteration {iteration} of repeat times {repeat_idx}")
            input("Rest manually!!!")

        # if env.use_gripper:
        #     env.replace_to_spatula()
        # push_start_list = [[73, 141], [72, 139], [99, 132], [92, 126]]
        # push_end_list = [[74, 104], [108, 129], [136, 130], [94, 89]]
        # for idx in range(len(push_start_list)):
        #     color_image, depth_image = utils.get_real_heightmap(env)
        #     if idx > 0:
        #         color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        #         cv2.arrowedLine(
        #             color_image,
        #             (int(push_start_list[idx-1][1]), int(push_start_list[idx-1][0])),
        #             (int(push_end_list[idx-1][1]), int(push_end_list[idx-1][0])),
        #             (255, 0, 255),
        #             2,
        #             tipLength=0.4,
        #         )
        #         color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        #     collector.save_heightmaps(iteration+idx, color_image, depth_image)
        #     push_start = [
        #         push_start_list[idx][0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
        #         push_start_list[idx][1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
        #         0.01,
        #     ]
        #     push_end = [
        #         push_end_list[idx][0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
        #         push_end_list[idx][1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
        #         0.01,
        #     ]
        #     env.push(push_start, push_end)
        # color_image, depth_image = utils.get_real_heightmap(env)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        # cv2.arrowedLine(
        #     color_image,
        #     (int(push_start_list[idx][1]), int(push_start_list[idx][0])),
        #     (int(push_end_list[idx][1]), int(push_end_list[idx][0])),
        #     (255, 0, 255),
        #     2,
        #     tipLength=0.4,
        # )
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # collector.save_heightmaps(iteration+idx+1, color_image, depth_image)
        # exit()

        # if env.use_gripper:
        #     env.replace_to_spatula()
        # push_start = [0.63, 0.038000000000000006, 0.01]
        # push_end = [0.55, -0.07800000000000001, 0.01]
        # env.push(push_start, push_end)
        # push_start = [0.5700000000000001, -0.032, 0.01]
        # push_end = [0.43800000000000006, -0.04600000000000001, 0.01]
        # env.push(push_start, push_end)

        # color_image = cv2.imread(
        #     "tree_plot/root.0-10_23_46_31.png"
        # )
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # depth_image = cv2.imread(
        #     "tree_plot/root.0-10_23_46_31-depth.png",
        #     cv2.IMREAD_UNCHANGED,
        # )
        # depth_image = depth_image.astype(np.float32) / 100000
        # q_value, best_pix_ind, grasp_predictions = predictor.get_grasp_q(
        #     color_image, depth_image, post_checking=True
        # )
        # print(f"Max grasp Q value: {q_value}")
        # grasp_pred_vis = predictor.get_prediction_vis(grasp_predictions, color_image, best_pix_ind)
        # cv2.imshow("test", grasp_pred_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        # color_image = cv2.imread(
        #     "tree_plot/root.0-169_122_146_93.png"
        # )
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # depth_image = cv2.imread(
        #     "tree_plot/root.0-169_122_146_93-depth.png",
        #     cv2.IMREAD_UNCHANGED,
        # )
        # depth_image = depth_image.astype(np.float32) / 100000
        # mask_infos = predictor.from_maskrcnn(color_image, depth_image, plot=True)
        # predictor.sample_actions(color_image, depth_image, mask_infos, plot=True)
        # exit()

        is_plot = False
        while True:
            if not IS_REAL:
                color_image, depth_image, _ = utils.get_true_heightmap(env)
            else:
                color_image, depth_image = utils.get_real_heightmap(env)
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            print(f"Target on the table (value: {np.sum(mask) / 255}) at iteration {iteration}")
            if np.sum(mask) / 255 < 10:
                break

            q_value, best_pix_ind, grasp_predictions = predictor.get_grasp_q(
                color_image, depth_image, post_checking=True
            )
            print(f"Max grasp Q value: {q_value}")

            # record
            collector.save_heightmaps(iteration, color_image, depth_image)
            grasp_pred_vis = predictor.get_prediction_vis(
                grasp_predictions, color_image, best_pix_ind
            )
            collector.save_visualizations(iteration, grasp_pred_vis, "grasp")

            # Grasp >>>>>
            if q_value > GRASP_Q_GRASP_THRESHOLD:
                best_rotation_angle = np.deg2rad(best_pix_ind[0] * (360.0 / NUM_ROTATION))
                primitive_position = [
                    best_pix_ind[1] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                    best_pix_ind[2] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                    depth_image[best_pix_ind[1]][best_pix_ind[2]] + WORKSPACE_LIMITS[2][0],
                ]
                if not IS_REAL:
                    if not env.use_gripper:
                        env.replace_to_gripper()
                    success, grasp_sucess = env.grasp(primitive_position, best_rotation_angle)
                else:
                    grasp_sucess = env.grasp(primitive_position, best_rotation_angle)
                    success = grasp_sucess
                # record
                reward_value = 1 if success and grasp_sucess else 0
                collector.executed_action_log.append(
                    [
                        1,  # grasp
                        primitive_position[0],
                        primitive_position[1],
                        primitive_position[2],
                        best_rotation_angle,
                        -1,
                        -1,
                    ]
                )
                collector.label_value_log.append(reward_value)
                collector.write_to_log("executed-action", collector.executed_action_log)
                collector.write_to_log("label-value", collector.label_value_log)
                iteration += 1
                if grasp_sucess:
                    break
                else:
                    continue
            # Grasp <<<<<

            # Search >>>>>
            mask_infos = predictor.from_maskrcnn(color_image, depth_image, plot=True)
            initial_state = PushState(
                "root",
                color_image,
                depth_image,
                q_value,
                0,
                mask_infos,
                predictor,
                max_q=GRASP_Q_PUSH_THRESHOLD,
                max_level=MCTS_MAX_LEVEL,
            )
            root = PushSearchNode(initial_state)
            mcts = MonteCarloTreeSearch(root)
            best_node = mcts.best_action(MCTS_ROLLOUTS)
            consecutive = False
            if len(best_node.children) > 0:
                best_child_q = 0
                best_child_idx = 0
                for idx, child in enumerate(best_node.children):
                    child_q = sum(sorted(child.q)[-MCTS_TOP:]) / min(child.n, MCTS_TOP)
                    if not is_consecutive(best_node.state, child.state):
                        child_q *= MCTS_DISCOUNT
                    if child_q > best_child_q:
                        best_child_q = child_q
                        best_child_idx = idx
                best_child_node = best_node.children[best_child_idx]
                if is_consecutive(best_node.state, best_child_node.state):
                    consecutive = True
            print("best node:")
            print(best_node.state.uid)
            print(sum(sorted(best_node.q)[-MCTS_TOP:]) / min(best_node.n, MCTS_TOP))
            print(best_node.state.q_value)
            print(best_node.prev_move)
            print(len(root.children))
            if consecutive:
                node = best_child_node
            else:
                node = best_node
            node_image = cv2.cvtColor(node.state.color_image, cv2.COLOR_RGB2BGR)
            node_action = str(node.prev_move).split("_")
            cv2.arrowedLine(
                node_image,
                (int(node_action[1]), int(node_action[0])),
                (int(node_action[3]), int(node_action[2])),
                (255, 0, 255),
                2,
                tipLength=0.4,
            )
            collector.save_predictions(iteration, node_image)
            # Search <<<<<

            # Push >>>>>
            push_start = best_node.prev_move.pose0
            if consecutive:
                push_end = best_child_node.prev_move.pose1
            else:
                push_end = best_node.prev_move.pose1
            push_start = [
                push_start[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                push_start[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                0.01,
            ]
            push_end = [
                push_end[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                push_end[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                0.01,
            ]
            if not IS_REAL:
                if env.use_gripper:
                    env.replace_to_spatula()
                env.push(push_start, push_end)
            else:
                env.push(push_start, push_end)
            # record
            reward_value = 0
            consecutive_value = 1 if consecutive else 0
            collector.executed_action_log.append(
                [
                    0,  # push
                    push_start[0],
                    push_start[1],
                    push_start[2],
                    push_end[0],
                    push_end[1],
                    push_end[2],
                ]
            )
            collector.label_value_log.append(reward_value)
            collector.consecutive_log.append(consecutive_value)
            collector.write_to_log("executed-action", collector.executed_action_log)
            collector.write_to_log("label-value", collector.label_value_log)
            collector.write_to_log("consecutive-value", collector.consecutive_log)
            iteration += 1
            # Push <<<<<

            # Plot
            if is_plot:
                files = glob.glob("tree_plot/*")
                for f in files:
                    os.remove(f)
                dot = Digraph(
                    "mcts",
                    filename="tree_plot/mcts.gv",
                    node_attr={
                        "shape": "box",
                        "fontcolor": "white",
                        "fontsize": "3",
                        "labelloc": "b",
                        "fixedsize": "true",
                    },
                )
                search_list = [root]
                while len(search_list) > 0:
                    current_node = search_list.pop(0)
                    node_name = current_node.state.uid
                    node_name_label = f"Q: {(sum(sorted(current_node.q)[-MCTS_TOP:]) / min(current_node.n, MCTS_TOP)):.3f},  N: {current_node.n},  Grasp Q: {current_node.state.q_value:.3f}"
                    node_image = cv2.cvtColor(current_node.state.color_image, cv2.COLOR_RGB2BGR)
                    if current_node.prev_move is not None:
                        node_action = str(current_node.prev_move).split("_")
                        cv2.arrowedLine(
                            node_image,
                            (int(node_action[1]), int(node_action[0])),
                            (int(node_action[3]), int(node_action[2])),
                            (255, 0, 255),
                            2,
                            tipLength=0.4,
                        )
                    image_name = f"tree_plot/{node_name}.png"
                    cv2.imwrite(image_name, node_image)
                    depthimage_name = f"tree_plot/{node_name}-depth.png"
                    generated_depth_image = current_node.state.depth_image
                    generated_depth_image = np.round(generated_depth_image * 100000).astype(
                        np.uint16
                    )  # Save depth in 1e-5 meters
                    cv2.imwrite(depthimage_name, generated_depth_image)
                    image_name = f"{node_name}.png"
                    image_size = str(
                        max(
                            0.6,
                            sum(sorted(current_node.q)[-MCTS_TOP:])
                            / min(current_node.n, MCTS_TOP)
                            * 2,
                        )
                    )
                    dot.node(
                        node_name,
                        label=node_name_label,
                        image=image_name,
                        width=image_size,
                        height=image_size,
                    )
                    if current_node.parent is not None:
                        node_partent_name = current_node.parent.state.uid
                        dot.edge(node_partent_name, node_name)
                    untracked_states = [current_node.state]
                    last_node_used = False
                    while len(untracked_states) > 0:
                        current_state = untracked_states.pop()
                        last_state_name = current_state.uid
                        if last_node_used:
                            actions = current_state.get_actions()
                        else:
                            if len(current_node.children) == 0:
                                actions = current_state.get_actions()
                            else:
                                actions = current_node.untried_actions
                            last_node_used = True
                        for idx, move in enumerate(actions):
                            key = current_state.uid + str(move)
                            if key in current_state.predictor.prediction_recorder:
                                (
                                    generated_color_image,
                                    generated_depth_image,
                                    _,
                                    new_image_q,
                                ) = current_state.predictor.prediction_recorder[key]
                                node_name = f"{current_state.uid}.{current_state.level}-{move}"
                                node_name_label = f"Grasp Q: {new_image_q:.3f}"
                                node_image = cv2.cvtColor(generated_color_image, cv2.COLOR_RGB2BGR)
                                node_action = str(move).split("_")
                                if len(node_action) > 1:
                                    cv2.arrowedLine(
                                        node_image,
                                        (int(node_action[1]), int(node_action[0])),
                                        (int(node_action[3]), int(node_action[2])),
                                        (255, 0, 255),
                                        2,
                                        tipLength=0.4,
                                    )
                                image_name = f"tree_plot/{node_name}.png"
                                cv2.imwrite(image_name, node_image)
                                depthimage_name = f"tree_plot/{node_name}-depth.png"
                                generated_depth_image = np.round(
                                    generated_depth_image * 100000
                                ).astype(
                                    np.uint16
                                )  # Save depth in 1e-5 meters
                                cv2.imwrite(depthimage_name, generated_depth_image)
                                image_name = f"{node_name}.png"
                                image_size = str(max(0.6, new_image_q * 2))
                                dot.node(
                                    node_name,
                                    label=node_name_label,
                                    image=image_name,
                                    width=image_size,
                                    height=image_size,
                                )
                                dot.edge(last_state_name, node_name)
                                new_state = current_state.move(move)
                                if new_state is not None:
                                    untracked_states.append(new_state)
                    search_list.extend(current_node.children)
                dot.view()
                input("wait for key")

            # clean up for memory
            del initial_state
            del mcts
            del root
            del best_node
            del push_start
            del push_end
            if consecutive:
                del best_child_node
            gc.collect()
            predictor.reset()
            gc.collect()