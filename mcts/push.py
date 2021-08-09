"""Class for MCTS."""

import math
import numpy as np
import cv2
import gc
from memory_profiler import profile
import sys

from constants import (
    TARGET_LOWER,
    TARGET_UPPER,
    PUSH_DISTANCE,
    PIXEL_SIZE,
    CONSECUTIVE_ANGLE_THRESHOLD,
    CONSECUTIVE_DISTANCE_THRESHOLD,
    MCTS_MAX_LEVEL,
    GRASP_Q_PUSH_THRESHOLD,
    BG_THRESHOLD,
    IMAGE_SIZE,
)


def is_consecutive(parent_state, current_state):
    if (
        parent_state.level > 0
        and abs(current_state.prev_angle - parent_state.prev_angle) < CONSECUTIVE_ANGLE_THRESHOLD
    ):
        this_move = current_state.uid.split("-")[-1].split("_")
        this_move = (int(this_move[0]), int(this_move[1]))
        prev_move = parent_state.uid.split("-")[-1].split("_")
        prev_move = (int(prev_move[2]), int(prev_move[3]))
        dis = math.sqrt((this_move[0] - prev_move[0]) ** 2 + (this_move[1] - prev_move[1]) ** 2)
        if dis < CONSECUTIVE_DISTANCE_THRESHOLD / PIXEL_SIZE:  # less than 5 cm
            return True
    return False


class PushMove:
    """Represent a move from start to end pose"""

    def __init__(self, pose0, pose1, precomputed_info):
        self.pose0 = pose0
        self.pose1 = pose1
        self.precomputed_info = precomputed_info

    def __str__(self):
        return f"{self.pose0[0]}_{self.pose0[1]}_{self.pose1[0]}_{self.pose1[1]}"

    def __repr__(self):
        return f"start: {self.pose0} to: {self.pose1}"

    def __eq__(self, other):
        return self.pose0 == other.pose0 and self.pose1 == other.pose1

    def __hash__(self):
        return hash((self.pose0, self.pose1))


class PushState:
    """Use move_recorder and prediction_recorder from predictor.
    move_recorder, Key is uid: '(key of parrent) + (level)'.
    prediction_recorder, Key is the uid: '(key of parrent) + (level)' + (move).
    """

    # TODO: how to get a good max_q, which could be used to decide an object is graspable
    def __init__(
        self,
        uid,
        color_image,
        depth_image,
        q_value,
        level,
        obj_masks,
        predictor,
        max_q=GRASP_Q_PUSH_THRESHOLD,
        max_level=MCTS_MAX_LEVEL,
        prev_angle=None,
        prev_move=None,
    ):
        self.uid = uid
        self.color_image = color_image
        self.depth_image = depth_image
        self.q_value = q_value
        self.level = level
        self.obj_masks = obj_masks
        self.predictor = predictor
        self.max_q = max_q
        self.max_level = max_level
        self.prev_angle = prev_angle
        self.prev_move = prev_move

    @property
    def push_result(self):
        """Return the grasp q value"""
        return self.q_value

    def is_push_over(self):
        """Should stop the search"""
        # if reaches the last defined level or the object can be grasp
        if self.level == self.max_level or self.q_value > self.max_q:
            return True

        # if no legal actions
        if self.uid in self.predictor.move_recorder:
            if len(self.predictor.move_recorder[self.uid]) == 0:
                return True

        # if not over - no result
        return False

    def move_result(self, move):
        """Return the result after a move"""
        key = self.uid + str(move)
        if key not in self.predictor.prediction_recorder:
            rotated_color_image = move.precomputed_info["rotated_color_image"]
            rotated_depth_image = move.precomputed_info["rotated_depth_image"]
            rotated_action = move.precomputed_info["rotated_action"]
            rotated_center = move.precomputed_info["rotated_center"]
            rotated_angle = move.precomputed_info["rotated_angle"]
            rotated_binary_objs = move.precomputed_info["rotated_binary_objs"]
            rotated_mask_objs = move.precomputed_info["rotated_mask_objs"]
            (
                generated_color_images,
                generated_depth_images,
                generated_obj_masks,
                validations,
            ) = self.predictor.predict(
                rotated_color_image,
                rotated_depth_image,
                rotated_action,
                rotated_center,
                rotated_angle,
                rotated_binary_objs,
                rotated_mask_objs,
                plot=False,
            )
            # if self.level == 0:
            #     for idx, img in enumerate(generated_color_images):
            #         overlay = self.color_image
            #         # added_image = cv2.addWeighted(generated_color_images[idx], 0.8, overlay, 0.4, 0)
            #         added_image = generated_color_images[idx].copy()
            #         img = cv2.cvtColor(added_image, cv2.COLOR_RGB2BGR)
            #         cv2.imshow('pre', img)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            assert len(generated_color_images) == 1
            if validations[0]:
                generated_color_image = generated_color_images[0]
                generated_depth_image = generated_depth_images[0]
                generated_obj_masks = generated_obj_masks[0]
                new_image_q, _, _ = self.predictor.get_grasp_q(
                    generated_color_image, generated_depth_image, post_checking=True
                )
                self.predictor.prediction_recorder[key] = (
                    generated_color_image,
                    generated_depth_image,
                    generated_obj_masks,
                    new_image_q,
                )
            else:
                return None
        else:
            (
                generated_color_image,
                generated_depth_image,
                generated_obj_masks,
                new_image_q,
            ) = self.predictor.prediction_recorder[key]

        return generated_color_image, generated_depth_image, generated_obj_masks, new_image_q

    def is_move_legal(self, new_image):
        # check if the target object is in the image
        temp = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        return np.sum(mask) / 255 > 50

    def move(self, move):
        result = self.move_result(move)
        if result is None:
            return None
        new_color_image, new_depth_image, new_obj_masks, new_image_q = result
        is_legal = self.is_move_legal(new_color_image)
        if not is_legal:
            return None
        push_angle = math.atan2(move.pose1[1] - move.pose0[1], move.pose1[0] - move.pose0[0])
        move_in_image = ((move.pose0[1], move.pose0[0]), (move.pose1[1], move.pose1[0]))
        return PushState(
            f"{self.uid}.{self.level}-{move}",
            new_color_image,
            new_depth_image,
            new_image_q,
            self.level + 1,
            new_obj_masks,
            self.predictor,
            max_q=self.max_q,
            max_level=self.max_level,
            prev_angle=push_angle,
            prev_move=move_in_image,
        )

    def get_actions(self):
        key = self.uid
        if key not in self.predictor.move_recorder:
            (
                rotated_color_image,
                rotated_depth_image,
                rotated_action,
                rotated_center,
                rotated_angle,
                rotated_binary_objs,
                before_rotated_action,
                rotated_mask_objs,
            ) = self.predictor.sample_actions(
                self.color_image,
                self.depth_image,
                self.obj_masks,
                plot=False,
                prev_move=self.prev_move,
            )
            moves = []
            for i in range(len(rotated_color_image)):
                angle = rotated_angle[i]
                if angle < 0:
                    angle = 360 + angle
                rotation_angle = np.deg2rad(angle)
                pose0 = (before_rotated_action[i][0], before_rotated_action[i][1])
                pose1 = [
                    round(pose0[0] + PUSH_DISTANCE / PIXEL_SIZE * np.cos(-rotation_angle)),
                    round(pose0[1] + PUSH_DISTANCE / PIXEL_SIZE * np.sin(-rotation_angle)),
                ]
                pose1 = (min(IMAGE_SIZE, max(0, pose1[0])), min(IMAGE_SIZE, max(0, pose1[1])))
                precomputed_info = {
                    "rotated_color_image": [rotated_color_image[i]],
                    "rotated_depth_image": [rotated_depth_image[i]],
                    "rotated_action": [rotated_action[i]],
                    "rotated_center": [rotated_center[i]],
                    "rotated_angle": [rotated_angle[i]],
                    "rotated_binary_objs": [rotated_binary_objs[i]],
                    "rotated_mask_objs": [rotated_mask_objs[i]],
                }
                moves.append(PushMove(pose0, pose1, precomputed_info))
            self.predictor.move_recorder[key] = moves
        else:
            moves = self.predictor.move_recorder[key]

        return moves

    def remove_action(self, move):
        key = self.uid
        if key in self.predictor.move_recorder:
            moves = self.predictor.move_recorder[key]
            if move in moves:
                moves.remove(move)
