import copy
import torch
import gc
import numpy as np
import cv2
from torchvision.transforms import functional as TF
import math

from push_net import PushPredictionNet
from models import reinforcement_net
from train_maskrcnn import get_model_instance_segmentation
from dataset import PushPredictionMultiDatasetEvaluation
from constants import (
    DEPTH_MIN,
    TARGET_LOWER,
    TARGET_UPPER,
    IS_REAL,
    IMAGE_SIZE,
    IMAGE_PAD_WIDTH,
    PUSH_DISTANCE,
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    NUM_ROTATION,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    GRIPPER_GRASP_SAFE_WIDTH_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    IMAGE_PAD_DIFF,
    PIXEL_SIZE,
    IMAGE_PAD_SIZE,
    PUSH_BUFFER,
    GRASP_Q_GRASP_THRESHOLD,
    BG_THRESHOLD,
    COLOR_SPACE
)
from action_utils_mask import sample_actions as sample_actions_util
import imutils
import utils


class PushPredictor:
    """
    Predict and generate images after push actions.
    Assume the color image and depth image are well matched.
    We use the masks to generate new images, so the quality of mask is important.
    The input to this forward function should be returned from the sample_actions.
    """

    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Initialize Mask R-CNN
        self.mask_model = get_model_instance_segmentation(2)
        self.mask_model.load_state_dict(torch.load("logs_image/maskrcnn.pth"))
        self.mask_model = self.mask_model.to(self.device)
        self.mask_model.eval()
        # Initialize Push Prediction
        self.push_model = PushPredictionNet()
        self.push_model.load_state_dict(
            torch.load("logs_push/push_prediction_model-75.pth")["model"]
        )
        self.push_model = self.push_model.to(self.device)
        self.push_model.eval()
        # Initialize Grasp Q Evaluation
        self.grasp_model = reinforcement_net()
        self.grasp_model.load_state_dict(
            torch.load("logs_grasp/snapshot-post-020000.reinforcement.pth")["model"]
        )
        self.grasp_model = self.grasp_model.to(self.device)
        self.grasp_model.eval()
        self.move_recorder = {}
        self.prediction_recorder = {}

    def reset(self):
        del self.move_recorder
        del self.prediction_recorder
        gc.collect()
        self.move_recorder = {}
        self.prediction_recorder = {}

    @torch.no_grad()
    def from_maskrcnn(self, color_image, depth_image, plot=False):
        """
        Use Mask R-CNN to do instance segmentation and output masks in binary format.
        """
        image = color_image.copy()
        image = TF.to_tensor(image)
        prediction = self.mask_model([image.to(self.device)])[0]
        mask_objs = []
        centers = []
        blue_idx = -1
        if plot:
            pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for idx, mask in enumerate(prediction["masks"]):
            # TODO, 0.9 can be tuned
            if IS_REAL:
                threshold = 0.97
            else:
                threshold = 0.98
            if prediction["scores"][idx] > threshold:
                # get mask
                img = mask[0].mul(255).byte().cpu().numpy()
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                if np.sum(img == 255) < 100:
                    continue
                # get center
                obj_cnt = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                obj_cnt = imutils.grab_contours(obj_cnt)
                obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))[
                    -1
                ]  # the mask r cnn could give bad masks
                M = cv2.moments(obj_cnt)
                cX = round(M["m10"] / M["m00"])
                cY = round(M["m01"] / M["m00"])
                # get color and depth masks
                color_mask = cv2.bitwise_and(color_image, color_image, mask=img)
                temp = cv2.cvtColor(color_mask, cv2.COLOR_RGB2HSV)
                temp = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
                if np.sum(temp == 255) >= 100:
                    blue_idx = idx
                depth_mask = cv2.bitwise_and(depth_image, depth_image, mask=img)
                # get cropped masks
                color_mask = np.pad(
                    color_mask,
                    (
                        (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH),
                        (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH),
                        (0, 0),
                    ),
                    "constant",
                    constant_values=0,
                )
                color_mask = color_mask[
                    cY + IMAGE_PAD_WIDTH - 30 : cY + IMAGE_PAD_WIDTH + 30,
                    cX + IMAGE_PAD_WIDTH - 30 : cX + IMAGE_PAD_WIDTH + 30,
                    :,
                ]
                depth_mask = np.pad(
                    depth_mask,
                    (
                        (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH),
                        (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH),
                    ),
                    "constant",
                    constant_values=0,
                )
                depth_mask = depth_mask[
                    cY + IMAGE_PAD_WIDTH - 30 : cY + IMAGE_PAD_WIDTH + 30,
                    cX + IMAGE_PAD_WIDTH - 30 : cX + IMAGE_PAD_WIDTH + 30,
                ]
                final_mask = (color_mask, depth_mask)
                mask_objs.append(final_mask)
                centers.append([cY + IMAGE_PAD_WIDTH, cX + IMAGE_PAD_WIDTH])
                if plot:
                    pred_mask[img > 0] = 255 - idx * 20
                    cv2.imwrite(str(idx) + "mask.png", img)
        if plot:
            cv2.imwrite("pred.png", pred_mask)
        print("Mask R-CNN: %d objects detected" % len(mask_objs), prediction["scores"].cpu())
        if blue_idx != -1 and blue_idx != 0:
            temp = mask_objs[0]
            mask_objs[0] = mask_objs[blue_idx]
            mask_objs[blue_idx] = temp
            temp = centers[0]
            centers[0] = centers[blue_idx]
            centers[blue_idx] = temp
        return mask_objs, centers

    def sample_actions(
        self, color_image, depth_image, mask_objs, plot=False, start_pose=None, prev_move=None
    ):
        """
        Sample actions around the objects, from the boundary to the center.
        Assume there is no object in "black"
        Output the rotated image, such that the push action is from left to right
        """
        return sample_actions_util(
            color_image,
            depth_image,
            mask_objs,
            plot,
            start_pose,
            from_color=True,
            prev_move=prev_move,
        )

    # only rotated_color_image, rotated_depth_image are padding to 320x320
    @torch.no_grad()
    def predict(
        self,
        rotated_color_image,
        rotated_depth_image,
        rotated_action,
        rotated_center,
        rotated_angle,
        rotated_binary_objs,
        rotated_mask_objs,
        plot=False,
    ):
        # get data
        dataset = PushPredictionMultiDatasetEvaluation(
            rotated_depth_image, rotated_action, rotated_center, rotated_binary_objs
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(rotated_depth_image), shuffle=False, num_workers=0
        )
        (
            prev_poses,
            action,
            action_start_ori,
            action_end_ori,
            used_binary_img,
            binary_objs_total,
            num_obj,
        ) = next(iter(data_loader))
        prev_poses = prev_poses.to(self.device, non_blocking=True)
        used_binary_img = used_binary_img.to(self.device, non_blocking=True, dtype=torch.float)
        binary_objs_total = binary_objs_total.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)
        # get output
        output = self.push_model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
        output = output.cpu().numpy()

        # generate new images
        prev_poses_input = prev_poses.cpu().numpy().astype(int)
        prev_poses = copy.deepcopy(prev_poses_input)
        action_start_ori = action_start_ori.numpy().astype(int)
        action_end_ori = action_end_ori.numpy().astype(int)
        action_start_ori_tile = np.tile(action_start_ori, num_obj[0])
        action_start = action[:, :2].cpu().numpy().astype(int)
        action_start_tile = np.tile(action_start, num_obj[0])
        generated_color_images = []
        generated_depth_images = []
        generated_obj_masks = []
        validations = []

        for i in range(len(rotated_depth_image)):
            i_output = output[i]
            i_prev_poses = prev_poses[i]
            i_action_start_ori_tile = action_start_ori_tile[i]
            i_action_start_tile = action_start_tile[i]
            i_prev_poses += i_action_start_ori_tile
            i_prev_poses -= i_action_start_tile
            i_rotated_angle = rotated_angle[i]
            i_rotated_mask_objs, i_rotated_mask_obj_centers = rotated_mask_objs[i]
            color_image = rotated_color_image[i]
            depth_image = rotated_depth_image[i]
            # transform points and fill them into a black image
            generated_color_image = np.zeros_like(color_image)
            generated_depth_image = np.zeros_like(depth_image)
            obj_masks = []
            obj_mask_centers = []
            temp_obj_masks = []
            temp_obj_mask_centers = []
            # for each object
            valid = True
            for pi in range(num_obj[i]):
                # if the object is out of the boundary, then, we can skip this action
                center = i_rotated_mask_obj_centers[pi]
                center = np.array([[center]])
                M = cv2.getRotationMatrix2D(
                    (
                        i_prev_poses[pi * 2] + IMAGE_PAD_WIDTH,
                        i_prev_poses[pi * 2 + 1] + IMAGE_PAD_WIDTH,
                    ),
                    -i_output[pi * 3 + 2],
                    1,
                )
                ori_M = M.copy()
                M[0, 2] += i_output[pi * 3]
                M[1, 2] += i_output[pi * 3 + 1]
                new_center = cv2.transform(center, M)
                new_center = np.transpose(new_center[0])
                temp_obj_mask_centers.append(new_center)
                ori_center = cv2.transform(center, ori_M)
                ori_center = np.transpose(ori_center[0])
                M = cv2.getRotationMatrix2D(
                    (IMAGE_PAD_SIZE // 2, IMAGE_PAD_SIZE // 2),
                    i_rotated_angle,
                    1,
                )
                new_center = [new_center[0][0], new_center[1][0]]
                new_center = np.array([[new_center]])
                new_center = cv2.transform(new_center, M)[0][0]
                obj_mask_centers.append(new_center)
                ori_center = [ori_center[0][0], ori_center[1][0]]
                ori_center = np.array([[ori_center]])
                ori_center = cv2.transform(ori_center, M)[0][0]
                if (
                    new_center[1] - IMAGE_PAD_WIDTH
                    > IMAGE_SIZE - PUSH_BUFFER / PIXEL_SIZE
                    or new_center[1] - IMAGE_PAD_WIDTH < PUSH_BUFFER / PIXEL_SIZE
                    or new_center[0] - IMAGE_PAD_WIDTH
                    > IMAGE_SIZE - PUSH_BUFFER / PIXEL_SIZE
                    or new_center[0] - IMAGE_PAD_WIDTH < PUSH_BUFFER / PIXEL_SIZE
                ): 
                    if not (
                        ori_center[1] - IMAGE_PAD_WIDTH
                        > IMAGE_SIZE - PUSH_BUFFER / PIXEL_SIZE
                        or ori_center[1] - IMAGE_PAD_WIDTH < PUSH_BUFFER / PIXEL_SIZE
                        or ori_center[0] - IMAGE_PAD_WIDTH
                        > IMAGE_SIZE - PUSH_BUFFER / PIXEL_SIZE
                        or ori_center[0] - IMAGE_PAD_WIDTH < PUSH_BUFFER / PIXEL_SIZE
                    ):
                        valid = False
                        break
            validations.append(valid)
            if valid:
                for pi in range(num_obj[i]):
                    # if the object is out of the boundary, then, we can skip this action
                    # if (
                    #     i_prev_poses[pi * 2 + 1] + i_output[pi * 3 + 1]
                    #     > IMAGE_SIZE - PUSH_BUFFER / PIXEL_SIZE
                    #     or i_prev_poses[pi * 2 + 1] + i_output[pi * 3 + 1] < PUSH_BUFFER / PIXEL_SIZE
                    #     or i_prev_poses[pi * 2] + i_output[pi * 3]
                    #     > IMAGE_SIZE - PUSH_BUFFER / PIXEL_SIZE
                    #     or i_prev_poses[pi * 2] + i_output[pi * 3] < PUSH_BUFFER / PIXEL_SIZE
                    # ): 
                    #     valid = False
                    #     break
                    # find out transformation
                    # mask
                    mask_color = i_rotated_mask_objs[pi][0]
                    mask_depth = i_rotated_mask_objs[pi][1]
                    rotated_color = utils.rotate(mask_color, i_output[pi * 3 + 2])
                    rotated_depth = utils.rotate(mask_depth, i_output[pi * 3 + 2])
                    temp_obj_masks.append((rotated_color, rotated_depth))
                    # center
                    # center = i_rotated_mask_obj_centers[pi]
                    # center = np.array([[center]])
                    # M = cv2.getRotationMatrix2D(
                    #     (
                    #         i_prev_poses[pi * 2] + IMAGE_PAD_WIDTH,
                    #         i_prev_poses[pi * 2 + 1] + IMAGE_PAD_WIDTH,
                    #     ),
                    #     -i_output[pi * 3 + 2],
                    #     1,
                    # )
                    # M[0, 2] += i_output[pi * 3]
                    # M[1, 2] += i_output[pi * 3 + 1]
                    # new_center = cv2.transform(center, M)
                    # new_center = np.transpose(new_center[0])
                    # temp_obj_mask_centers.append(new_center)
                # validations.append(valid)
                # if valid:
                for pi in range(num_obj[i]):
                    mask = temp_obj_masks[pi]
                    new_center = temp_obj_mask_centers[pi]
                    color = mask[0]
                    fill_color = np.nonzero(np.sum(color, axis=2))
                    fill_color_shift = (
                        np.clip(fill_color[0] + new_center[0] - 30, 0, IMAGE_PAD_SIZE - 1),
                        np.clip(fill_color[1] + new_center[1] - 30, 0, IMAGE_PAD_SIZE - 1)
                    )
                    generated_color_image[fill_color_shift] = color[fill_color]
                    depth = mask[1]
                    fill_depth = np.nonzero(depth)
                    fill_depth_shift = (
                        np.clip(fill_depth[0] + new_center[0] - 30, 0, IMAGE_PAD_SIZE - 1),
                        np.clip(fill_depth[1] + new_center[1] - 30, 0, IMAGE_PAD_SIZE - 1)
                    )
                    generated_depth_image[fill_depth_shift] = depth[fill_depth]
                    generated_obj_mask_color = utils.rotate(color, -i_rotated_angle)
                    generated_obj_mask_depth = utils.rotate(depth, -i_rotated_angle)
                    obj_masks.append((generated_obj_mask_color, generated_obj_mask_depth))
                    # M = cv2.getRotationMatrix2D(
                    #     (IMAGE_PAD_SIZE // 2, IMAGE_PAD_SIZE // 2),
                    #     i_rotated_angle,
                    #     1,
                    # )
                    # new_center = [new_center[0][0], new_center[1][0]]
                    # new_center = np.array([[new_center]])
                    # new_center = cv2.transform(new_center, M)[0][0]
                    # obj_mask_centers.append(new_center)
                    if plot:
                        cv2.circle(
                            generated_color_image,
                            (
                                i_prev_poses[pi * 2 + 1] + IMAGE_PAD_WIDTH,
                                i_prev_poses[pi * 2] + IMAGE_PAD_WIDTH,
                            ),
                            3,
                            (255, 255, 255),
                            -1,
                        )
                        cv2.circle(
                            generated_color_image,
                            (
                                round(i_prev_poses[pi * 2 + 1] + i_output[pi * 3 + 1]) + IMAGE_PAD_WIDTH,
                                round(i_prev_poses[pi * 2] + i_output[pi * 3]) + IMAGE_PAD_WIDTH,
                            ),
                            3,
                            (128, 255, 0),
                            -1,
                        )
                if plot:
                    cv2.arrowedLine(
                        generated_color_image,
                        (
                            action_start_ori[i][1] + IMAGE_PAD_WIDTH,
                            action_start_ori[i][0] + IMAGE_PAD_WIDTH,
                        ),
                        (
                            action_end_ori[i][1] + IMAGE_PAD_WIDTH,
                            action_end_ori[i][0] + IMAGE_PAD_WIDTH,
                        ),
                        (255, 0, 255),
                        2,
                        tipLength=0.4,
                    )
                generated_color_image = utils.rotate(generated_color_image, -i_rotated_angle)
                generated_depth_image = utils.rotate(generated_depth_image, -i_rotated_angle)
                generated_color_image = generated_color_image[
                    IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, :
                ]
                generated_depth_image = generated_depth_image[
                    IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF
                ]
                generated_depth_image = generated_depth_image.astype(np.float32)

            generated_color_images.append(generated_color_image)
            generated_depth_images.append(generated_depth_image)
            generated_obj_masks.append((obj_masks, obj_mask_centers))

        return generated_color_images, generated_depth_images, generated_obj_masks, validations

    @torch.no_grad()
    def get_grasp_q(self, color_heightmap, depth_heightmap, post_checking=False):
        color_heightmap_pad = np.copy(color_heightmap)
        depth_heightmap_pad = np.copy(depth_heightmap)

        # use light color
        if IS_REAL:
            temp = cv2.cvtColor(color_heightmap_pad, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            color_heightmap_pad[mask == 255] = [118, 183, 178]  # cyan

        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color_heightmap_pad,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(
            depth_heightmap_pad, IMAGE_PAD_WIDTH, "constant", constant_values=0
        )

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
            1,
        )
        input_depth_image.shape = (
            input_depth_image.shape[0],
            input_depth_image.shape[1],
            input_depth_image.shape[2],
            1,
        )
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(
            3, 2, 0, 1
        )

        # Pass input data through model
        output_prob = self.grasp_model(input_color_data, input_depth_data, True, -1, False)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = (
                    output_prob[rotate_idx][1]
                    .cpu()
                    .data.numpy()[
                        :,
                        0,
                        :,
                        :,
                    ]
                )
            else:
                grasp_predictions = np.concatenate(
                    (
                        grasp_predictions,
                        output_prob[rotate_idx][1]
                        .cpu()
                        .data.numpy()[
                            :,
                            0,
                            :,
                            :,
                        ],
                    ),
                    axis=0,
                )

        # post process, only grasp one object, focus on blue object
        temp = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
        mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
        mask_bg = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
        mask_bg_pad = np.pad(mask_bg, IMAGE_PAD_WIDTH, "constant", constant_values=255)
        # focus on blue
        for rotate_idx in range(len(grasp_predictions)):
            grasp_predictions[rotate_idx][mask_pad != 255] = 0
        padding_width_start = IMAGE_PAD_WIDTH
        padding_width_end = grasp_predictions[0].shape[0] - IMAGE_PAD_WIDTH
        # only grasp one object
        kernel_big = np.ones(
            (GRIPPER_GRASP_SAFE_WIDTH_PIXEL, GRIPPER_GRASP_INNER_DISTANCE_PIXEL), dtype=np.uint8
        )
        if IS_REAL:  # due to color, depth sensor and lighting, the size of object looks a bit smaller.
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 7
            threshold_small = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 15
        else:
            threshold_big = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 10
            threshold_small = GRIPPER_GRASP_SAFE_WIDTH_PIXEL * GRIPPER_GRASP_INNER_DISTANCE_PIXEL / 20
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1])
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
            grasp_predictions[rotate_idx][invalid_mask > threshold_small] = (
                grasp_predictions[rotate_idx][invalid_mask > threshold_small] / 2
            )
            grasp_predictions[rotate_idx][invalid_mask > threshold_big] = 0

        # collision checking, only work for one level
        if post_checking:
            mask = cv2.inRange(temp, BG_THRESHOLD["low"], BG_THRESHOLD["high"])
            mask = 255 - mask
            mask_pad = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
            check_kernel = np.ones(
                (GRIPPER_GRASP_WIDTH_PIXEL, GRIPPER_GRASP_OUTER_DISTANCE_PIXEL), dtype=np.uint8
            )
            left_bound = math.floor(
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
                grasp_predictions[rotate_idx][invalid_mask > 128] = 0
        grasp_predictions = grasp_predictions[
            :, padding_width_start:padding_width_end, padding_width_start:padding_width_end
        ]

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        grasp_q_value = grasp_predictions[best_pix_ind]

        return grasp_q_value, best_pix_ind, grasp_predictions

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations / 4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row * 4 + canvas_col
                prediction_vis = predictions[rotate_idx, :, :].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap(
                    (prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(
                        prediction_vis,
                        (int(best_pix_ind[2]), int(best_pix_ind[1])),
                        7,
                        (0, 0, 255),
                        2,
                    )
                prediction_vis = utils.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations))
                if rotate_idx == best_pix_ind[0]:
                    center = np.array([[[int(best_pix_ind[2]), int(best_pix_ind[1])]]])
                    M = cv2.getRotationMatrix2D(
                        (
                            prediction_vis.shape[1] // 2,
                            prediction_vis.shape[0] // 2,
                        ),
                        rotate_idx * (360.0 / num_rotations),
                        1,
                    )
                    center = cv2.transform(center, M)
                    center = np.transpose(center[0])
                    prediction_vis = cv2.rectangle(
                        prediction_vis,
                        (max(0, int(center[0]) - GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2), max(0, int(center[1]) - GRIPPER_GRASP_WIDTH_PIXEL // 2)),
                        (min(prediction_vis.shape[1], int(center[0]) + GRIPPER_GRASP_INNER_DISTANCE_PIXEL // 2), min(prediction_vis.shape[0], int(center[1]) + GRIPPER_GRASP_WIDTH_PIXEL // 2)),
                        (100, 255, 0),
                        1
                    )
                    prediction_vis = cv2.rectangle(
                        prediction_vis,
                        (max(0, int(center[0])  - GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2), max(0, int(center[1]) - GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2)),
                        (min(prediction_vis.shape[1], int(center[0])  + GRIPPER_GRASP_OUTER_DISTANCE_PIXEL // 2), min(prediction_vis.shape[0], int(center[1]) + GRIPPER_GRASP_SAFE_WIDTH_PIXEL // 2)),
                        (100, 100, 155),
                        1,
                    )
                background_image = utils.rotate(
                    color_heightmap, rotate_idx * (360.0 / num_rotations)
                )
                prediction_vis = (
                    0.5 * cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis
                ).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

        return canvas
