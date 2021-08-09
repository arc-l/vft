import os
import time
import numpy as np
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import reinforcement_net, reactive_net
from scipy import ndimage
import matplotlib.pyplot as plt
from constants import (COLOR_MEAN, COLOR_STD, DEPTH_MEAN, DEPTH_STD, DEPTH_MIN, 
blue_lower, blue_upper, red_lower, red_upper, IMAGE_PAD_WIDTH, NUM_ROTATION, 
GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
TARGET_LOWER, TARGET_UPPER,GRASP_Q_GRASP_THRESHOLD,BG_THRESHOLD,GRIPPER_GRASP_WIDTH_PIXEL,
IMAGE_PAD_SIZE, IMAGE_PAD_DIFF, IMAGE_SIZE)
import utils


class Trainer(object):
    def __init__(self, method, push_rewards, future_reward_discount,
                 is_testing, load_snapshot, snapshot_file, force_cpu, is_baseline=False):

        self.method = method
        self.is_baseline = is_baseline

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional classification network for supervised learning
        if self.method == 'reinforcement':
            self.model = reinforcement_net()
            self.push_rewards = push_rewards
            self.future_reward_discount = future_reward_discount

            # Initialize Huber loss
            self.push_criterion = torch.nn.SmoothL1Loss(reduction='none')  # Huber loss
            self.grasp_criterion = torch.nn.SmoothL1Loss(reduction='none')  # Huber loss
            if self.use_cuda:
                self.push_criterion = self.push_criterion.cuda()
                self.grasp_criterion = self.grasp_criterion.cuda()

        # Load pre-trained model
        if load_snapshot:
            states = torch.load(snapshot_file)
            if "model" in states:
                self.model.load_state_dict(states["model"])
            else:
                self.model.load_state_dict(states)
            print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.iteration = 0
        if is_testing:
            if is_baseline:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-5)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-5)
        else:
            if is_baseline:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-5)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=5e-5, momentum=0.9, weight_decay=2e-5)
        if is_baseline:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.5)

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []
        self.loss_log = []

        if is_testing:
            self.batch_size = 2
        else:
            self.batch_size = 12
        self.loss_list = []

    # Pre-load execution info and RL variables

    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(
            os.path.join(
                transitions_directory,
                'executed-action.log.txt'),
            delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration, 1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(
            os.path.join(
                transitions_directory,
                'predicted-value.log.txt'),
            delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration, 1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration, 1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0], 1)
        self.clearance_log = self.clearance_log.tolist()

    # Compute forward pass through model to compute affordances/Q

    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1, use_push=True):

        color_heightmap_pad = np.copy(color_heightmap)
        depth_heightmap_pad = np.copy(depth_heightmap)

        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color_heightmap_pad,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(depth_heightmap_pad, IMAGE_PAD_WIDTH, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = COLOR_MEAN
        image_std = COLOR_STD
        input_color_image = color_heightmap_pad.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1], 1)
        input_depth_image = np.copy(depth_heightmap_pad)
        input_depth_image[:, :, 0] = (input_depth_image[:, :, 0] - image_mean[0]) / image_std[0]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0],
            input_color_image.shape[1],
            input_color_image.shape[2],
            1)
        input_depth_image.shape = (
            input_depth_image.shape[0],
            input_depth_image.shape[1],
            input_depth_image.shape[2],
            1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

        # Pass input data through model
        output_prob = self.model(input_color_data, input_depth_data, is_volatile, specific_rotation, use_push)

        if self.method == 'reinforcement':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    if not use_push:
                        push_predictions = 0
                        grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, int(IMAGE_PAD_WIDTH):int(
                            color_heightmap_pad.shape[0] - IMAGE_PAD_WIDTH), int(IMAGE_PAD_WIDTH):int(color_heightmap_pad.shape[1] - IMAGE_PAD_WIDTH)]
                    else:
                        push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:, 0, int(IMAGE_PAD_WIDTH):int(
                            color_heightmap_pad.shape[0] - IMAGE_PAD_WIDTH), int(IMAGE_PAD_WIDTH):int(color_heightmap_pad.shape[1] - IMAGE_PAD_WIDTH)]
                        grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:, 0, int(IMAGE_PAD_WIDTH):int(
                            color_heightmap_pad.shape[0] - IMAGE_PAD_WIDTH), int(IMAGE_PAD_WIDTH):int(color_heightmap_pad.shape[1] - IMAGE_PAD_WIDTH)]
                else:
                    if not use_push:
                        push_predictions = 0
                        grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[
                            :, 0, int(IMAGE_PAD_WIDTH):int(color_heightmap_pad.shape[0] - IMAGE_PAD_WIDTH), int(IMAGE_PAD_WIDTH):int(
                                color_heightmap_pad.shape[1] - IMAGE_PAD_WIDTH)]), axis=0)
                    else:
                        push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[
                            :, 0, int(IMAGE_PAD_WIDTH):int(color_heightmap_pad.shape[0] - IMAGE_PAD_WIDTH),
                            int(IMAGE_PAD_WIDTH):int(color_heightmap_pad.shape[1] - IMAGE_PAD_WIDTH)]), axis=0)
                        grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[
                            :, 0, int(IMAGE_PAD_WIDTH):int(color_heightmap_pad.shape[0] - IMAGE_PAD_WIDTH), int(IMAGE_PAD_WIDTH):int(
                                color_heightmap_pad.shape[1] - IMAGE_PAD_WIDTH)]), axis=0)

        return push_predictions, grasp_predictions

    def focus_on_target(self, color_heightmap, depth_heightmap, grasp_predictions, target_lower, target_upper):
        """Should match push_predictor"""
        # focus on target object
        temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
        # focus on blue
        for rotate_idx in range(len(grasp_predictions)):
            grasp_predictions[rotate_idx][mask != 255] = 0
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions[0].shape[0] + IMAGE_PAD_WIDTH
        # only grasp one object
        kernel_big = np.ones(
            (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.float32
        )
        threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
        threshold_small = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
        depth_heightmap_pad = np.pad(depth_heightmap, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        for rotate_idx in range(len(grasp_predictions)):
            color_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            color_mask[color_mask == 0] = 1
            color_mask[color_mask == 255] = 0
            no_target_mask = color_mask
            bg_mask = utils.rotate(mask_bg_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            no_target_mask[bg_mask == 255] = 0
            # only grasp one object
            invalid_mask = cv2.filter2D(no_target_mask, -1, kernel_big)
            invalid_mask = utils.rotate(invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True)
            invalid_mask = invalid_mask[padding_width_start:padding_width_end, padding_width_start:padding_width_end]
            grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
            )
            grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0
        
        # collision checking
        mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        mask = 255 - mask
        mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        check_kernel = np.ones(
            (GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8
        )
        left_bound = math.ceil(
            (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL - GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
        )
        right_bound = (
            math.ceil(
                (GRIPPER_GRASP_OUTER_DISTANCE_PIXEL + GRIPPER_GRASP_INNER_DISTANCE_PIXEL) / 2
            )
            + 1
        )
        check_kernel[:, left_bound:right_bound] = 0
        for rotate_idx in range(len(grasp_predictions)):
            object_mask = utils.rotate(mask_pad, rotate_idx * (360.0 / NUM_ROTATION), True)
            invalid_mask = cv2.filter2D(object_mask, -1, check_kernel)
            invalid_mask[invalid_mask > 5] = 255
            invalid_mask = utils.rotate(
                invalid_mask, -rotate_idx * (360.0 / NUM_ROTATION), True
            )
            invalid_mask = invalid_mask[
                padding_width_start:padding_width_end, padding_width_start:padding_width_end
            ]
            grasp_predictions[rotate_idx][invalid_mask > 128] = 0

        return grasp_predictions

    def get_label_value(self, primitive_action, push_success, grasp_success, change_detected, prev_push_predictions,
                        prev_grasp_predictions, next_color_heightmap, next_depth_heightmap, prev_depth_heightmap, use_push=True):

        if self.method == 'reinforcement':

            # Compute current reward
            current_reward = 0
            if primitive_action == 'push':
                if change_detected:
                    current_reward = 0.0
            elif primitive_action == 'grasp':
                if grasp_success:
                    current_reward = 1.0

            # Compute future reward
            if not change_detected and not grasp_success:
                future_reward = 0
            else:
                future_reward = 0  # no future reward
                if primitive_action == 'push':
                    _, next_grasp_predictions = self.forward(
                        next_color_heightmap, next_depth_heightmap, is_volatile=True, use_push=use_push)
                    if np.max(next_grasp_predictions) > np.max(prev_grasp_predictions) * 1.1:
                        current_reward = (np.max(next_grasp_predictions) + np.max(prev_grasp_predictions)) / 2
                    else:
                        future_reward = 0
                    print("Prediction:", np.max(prev_grasp_predictions), np.max(next_grasp_predictions))
                    delta_area = self.push_change_area(prev_depth_heightmap, next_depth_heightmap)
                    if delta_area > 300:  # 300 can be changed
                        if current_reward < 0.8:
                            current_reward = 0.8
                    elif delta_area < -100:  # -100 can be changed
                        current_reward = 0
                        future_reward = 0

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if primitive_action == 'push' and not self.push_rewards:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' %
                      (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print(
                    'Expected reward: %f + %f x %f = %f' %
                    (current_reward,
                     self.future_reward_discount,
                     future_reward,
                     expected_reward))
            return expected_reward, current_reward

    def get_label_value_base(self, primitive_action, push_success, grasp_success, change_detected, prev_push_predictions,
                        prev_grasp_predictions, next_color_heightmap, next_depth_heightmap, 
                        use_push=True, target=None, prev_color_img=None, prev_depth_img=None):
        """As baseline label value function"""

        if self.method == 'reinforcement':

            # Compute current reward
            current_reward = 0
            if primitive_action == 'push':
                if change_detected:
                    current_reward = 0.0
            elif primitive_action == 'grasp':
                if grasp_success:
                    crop = prev_color_img[max(0, target[0]-2) : min(target[0]+3, IMAGE_SIZE), max(0, target[1]-2) : min(target[1]+3, IMAGE_SIZE)]
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                    mask = cv2.inRange(crop, TARGET_LOWER, TARGET_UPPER)
                    if np.sum(mask) > 0:
                        current_reward = 20
                    else:
                        current_reward = 0

            # Compute future reward
            if not change_detected and not grasp_success:
                future_reward = 0
            else:
                next_push_predictions, next_grasp_predictions = self.forward(
                    next_color_heightmap, next_depth_heightmap, is_volatile=True, use_push=use_push)
                target_next_grasp_predictions = self.focus_on_target(next_color_heightmap, next_depth_heightmap, next_grasp_predictions, TARGET_LOWER, TARGET_UPPER)
                target_prev_grasp_predictions = self.focus_on_target(prev_color_img, prev_depth_img, prev_grasp_predictions, TARGET_LOWER, TARGET_UPPER)
                future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))
                if primitive_action == 'push':
                    if np.max(target_next_grasp_predictions) > np.max(target_prev_grasp_predictions) * 1.1:
                        current_reward = 1
                    print("Prediction:", np.max(prev_grasp_predictions), np.max(next_grasp_predictions))
                    delta_area = self.push_change_area(prev_depth_img, next_depth_heightmap)
                    if delta_area > 300:  # 300 can be changed
                        if current_reward < 1:
                            current_reward = 0.5
                    elif delta_area < -100:  # -100 can be changed
                        current_reward = 0
                        future_reward = 0

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if primitive_action == 'push' and not self.push_rewards:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' %
                      (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print(
                    'Expected reward: %f + %f x %f = %f' %
                    (current_reward,
                     self.future_reward_discount,
                     future_reward,
                     expected_reward))
            return expected_reward, current_reward

    def get_neg(self, depth_heightmap, label, best_pix_ind):
        """Should match train_foreground"""
        depth_heightmap_pad = np.copy(depth_heightmap)
        depth_heightmap_pad = np.pad(depth_heightmap_pad, IMAGE_PAD_WIDTH, 'constant', constant_values=0)
        depth_heightmap_pad = utils.rotate(depth_heightmap_pad, best_pix_ind * (360.0 / NUM_ROTATION))
        label = ndimage.rotate(label, best_pix_ind * (360.0 / NUM_ROTATION), axes=(2, 1), reshape=False)
        label = np.round(label)
        x_y_idx = np.argwhere(label > 0)
        for idx in x_y_idx:
            _, x, y = tuple(idx)
            left_area = depth_heightmap_pad[max(0, x - math.ceil(GRIPPER_GRASP_SAFE_WIDTH_PIXEL / 2)):min(depth_heightmap_pad.shape[0], x + math.ceil(GRIPPER_GRASP_SAFE_WIDTH_PIXEL / 2) + 1),
                                            max(0, y - math.ceil(GRIPPER_GRASP_OUTER_DISTANCE_PIXEL / 2)):max(0, y - math.ceil(GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 2) + 1)] 
            right_area = depth_heightmap_pad[max(0, x - math.ceil(GRIPPER_GRASP_SAFE_WIDTH_PIXEL / 2)):min(depth_heightmap_pad.shape[0], x + math.ceil(GRIPPER_GRASP_SAFE_WIDTH_PIXEL / 2) + 1),
                                             min(depth_heightmap_pad.shape[1] - 1, y + math.ceil(GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 2)):min(depth_heightmap_pad.shape[1], y + math.ceil(GRIPPER_GRASP_OUTER_DISTANCE_PIXEL / 2) + 1)]  
            if ((np.sum(left_area > DEPTH_MIN) > 0 and np.sum((left_area - depth_heightmap_pad[x, y]) > -0.05) > 0) or
                (np.sum(right_area > DEPTH_MIN) > 0 and np.sum((right_area - depth_heightmap_pad[x, y]) > -0.05) > 0)):
                label[0, x, y] = 0

        label = ndimage.rotate(label, -best_pix_ind * (360.0 / NUM_ROTATION), axes=(2, 1), reshape=False)
        label = np.round(label)
        return label

    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, use_push=True):

        if self.method == 'reinforcement':
            batch_lose = -1

            # Compute labels
            label = np.zeros((1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE))
            action_area = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            tmp_label = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            tmp_label[action_area > 0] = label_value
            label[0, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224, 224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF] = tmp_label_weights

            # Compute loss and backward pass
            if len(self.loss_list) == 0:
                self.optimizer.zero_grad()
            loss_value = 0

            if primitive_action == 'grasp' and label_value > 0:
                neg_loss = []
                for i in range(self.model.num_rotations):
                    if i != best_pix_ind[0]:
                        neg_label = self.get_neg(depth_heightmap, label.copy(), i)
                        if neg_label[0, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF][best_pix_ind[1]][best_pix_ind[2]] == 0:
                            _, _ = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=i, use_push=use_push)
                            loss = self.grasp_criterion(self.model.output_prob[0][1].view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), torch.from_numpy(neg_label).view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda()) * Variable(
                                torch.from_numpy(label_weights).view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda())
                            loss = loss.sum()
                            neg_loss.append(loss)
                if len(neg_loss) > 0:
                    self.loss_list.append(sum(neg_loss) / len(neg_loss))

            if primitive_action == 'push':
                if not self.is_baseline:
                    if label_value > 0:
                        label_weights *= 2  # to compromise the less push operations

                # Do forward pass with specified rotation (to save gradients)
                _, _ = self.forward(color_heightmap, depth_heightmap, is_volatile=False,
                                    specific_rotation=best_pix_ind[0], use_push=use_push)

                if self.use_cuda:
                    loss = self.push_criterion(
                        self.model.output_prob[0][0].view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), Variable(torch.from_numpy(label).view(
                                1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda())) * Variable(torch.from_numpy(label_weights).view(
                            1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda(), requires_grad=False)
                else:
                    loss = self.push_criterion(self.model.output_prob[0][0].view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), Variable(
                        torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(), requires_grad=False)
                loss = loss.sum()
                if len(self.loss_list) >= self.batch_size:
                    total_loss = sum(self.loss_list)
                    print('Batch Loss:', total_loss.cpu().item())
                    self.loss_log.append([self.iteration, total_loss.cpu()])
                    mean_loss = total_loss / len(self.loss_list)
                    mean_loss.backward()
                    self.loss_list = []
                else:
                    self.loss_list.append(loss)
                # loss.backward()
                loss_value = loss.cpu().data.numpy()

            elif primitive_action == 'grasp':
                if label_value > 0:
                    if self.is_baseline:
                        label_weights *= 4
                    else:
                        label_weights *= 2

                # Do forward pass with specified rotation (to save gradients)
                _, _ = self.forward(color_heightmap, depth_heightmap, is_volatile=False,
                                    specific_rotation=best_pix_ind[0], use_push=use_push)

                if self.use_cuda:
                    loss = self.grasp_criterion(
                        self.model.output_prob[0][1].view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), Variable(
                            torch.from_numpy(label).view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda())) * Variable(
                        torch.from_numpy(label_weights).view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda())
                else:
                    loss = self.grasp_criterion(self.model.output_prob[0][1].view(1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), Variable(
                        torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float())
                loss = loss.sum()
                self.loss_list.append(loss)
                # loss.backward()
                loss_value = loss.cpu().data.numpy()

                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations

                _, _ = self.forward(color_heightmap, depth_heightmap, is_volatile=False,
                                    specific_rotation=opposite_rotate_idx, use_push=use_push)

                if self.use_cuda:
                    loss = self.grasp_criterion(
                        self.model.output_prob[0][1].view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), Variable(
                            torch.from_numpy(label).view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda())) * Variable(
                        torch.from_numpy(label_weights).view(1, 1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE).float().cuda())
                else:
                    loss = self.grasp_criterion(self.model.output_prob[0][1].view(1, IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), Variable(
                        torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float())

                loss = loss.sum()
                if len(self.loss_list) >= self.batch_size:
                    total_loss = sum(self.loss_list)
                    print('Batch Loss:', total_loss.cpu().item())
                    batch_lose = total_loss.cpu().item()
                    self.loss_log.append([self.iteration, total_loss.cpu()])
                    mean_loss = total_loss / len(self.loss_list)
                    mean_loss.backward()
                    self.loss_list = []
                else:
                    self.loss_list.append(loss)
                # loss.backward()
                loss_value += loss.cpu().data.numpy()

                loss_value = loss_value / 2

            if len(self.loss_list) == 0:
                print('Training loss: %f' % (loss_value.sum()))
                self.optimizer.step()
                self.lr_scheduler.step()

            return batch_lose

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations / 4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row * 4 + canvas_col
                prediction_vis = predictions[rotate_idx, :, :].copy()
                # prediction_vis[prediction_vis < 0] = 0 # assume probability
                # prediction_vis[prediction_vis > 1] = 1 # assume probability
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(
                        prediction_vis, (int(
                            best_pix_ind[2]), int(
                            best_pix_ind[1])), 7, (0, 0, 255), 2)
                prediction_vis = utils.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations))
                background_image = utils.rotate(color_heightmap, rotate_idx * (360.0 / num_rotations))
                prediction_vis = (0.5 * cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

        return canvas

    def push_change_area(self, prev_depth_img, next_depth_img):

        kernel = np.ones((11, 11))
        # kernel_num = np.ones((5, 5))

        depth_img = np.copy(prev_depth_img)
        depth_img_copy = np.copy(depth_img)
        depth_img_copy[depth_img_copy <= DEPTH_MIN] = 0
        depth_img_copy[depth_img_copy > DEPTH_MIN] = 1
        prev_area = cv2.filter2D(depth_img_copy, -1, kernel)
        prev_area[prev_area <= 1] = 0
        prev_area[prev_area > 1] = 1
        prev_area = np.sum(prev_area - depth_img_copy)

        depth_img = np.copy(next_depth_img)
        depth_img_copy = np.copy(depth_img)
        depth_img_copy[depth_img_copy <= DEPTH_MIN] = 0
        depth_img_copy[depth_img_copy > DEPTH_MIN] = 1
        next_area = cv2.filter2D(depth_img_copy, -1, kernel)
        next_area[next_area <= 1] = 0
        next_area[next_area > 1] = 1
        next_area = np.sum(next_area - depth_img_copy)

        print("Prev Area %d" % (prev_area))
        print("Next Area %d" % (next_area))

        return next_area - prev_area