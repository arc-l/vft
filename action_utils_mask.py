import cv2
import imutils
import math
import random
from constants import (
    WORKSPACE_LIMITS,
    colors_lower,
    colors_upper,
    IMAGE_PAD_SIZE,
    IMAGE_SIZE,
    IMAGE_PAD_WIDTH,
    PUSH_DISTANCE,
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
    PIXEL_SIZE,
    DEPTH_MIN,
    IMAGE_SIZE,
    CONSECUTIVE_ANGLE_THRESHOLD,
    CONSECUTIVE_DISTANCE_THRESHOLD,
    IMAGE_PAD_WIDTH,
    TARGET_LOWER,
    TARGET_UPPER,
    PUSH_BUFFER,
    IMAGE_PAD_DIFF,
    GRIPPER_GRASP_WIDTH_PIXEL,
)
import numpy as np
import torch
from dataset import PushPredictionMultiDatasetEvaluation
from push_net import PushPredictionNet
from train_maskrcnn import get_model_instance_segmentation
from torchvision.transforms import functional as TF
import copy
import utils
from memory_profiler import profile
from trainer import Trainer


class Predictor:
    """
    Predict and generate images after push actions.
    Assume the color image and depth image are well matched.
    We use the masks to generate new images, so the quality of mask is important.
    The input to this forward function should be returned from the sample_actions.
    """

    def __init__(self, snapshot):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        push_model = PushPredictionNet()
        state = torch.load(snapshot)["model"]
        push_model.load_state_dict(state)
        self.push_model = push_model.to(self.device)
        self.push_model.eval()

    # only rotated_color_image, rotated_depth_image are padding to 320x320
    @torch.no_grad()
    def forward(
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
        validations = []

        for i in range(len(rotated_depth_image)):
            i_output = output[i]
            i_prev_poses = prev_poses[i]
            i_action_start_ori_tile = action_start_ori_tile[i]
            i_action_start_tile = action_start_tile[i]
            i_prev_poses += i_action_start_ori_tile
            i_prev_poses -= i_action_start_tile
            i_rotated_angle = rotated_angle[i]
            i_rotated_mask_objs = rotated_mask_objs[i]
            color_image = rotated_color_image[i]
            depth_image = rotated_depth_image[i]
            # transform points and fill them into a black image
            generated_color_image = np.zeros_like(color_image)
            generated_depth_image = np.zeros_like(depth_image)
            post_points_pad = []
            post_new_points_pad = []

            # for each object
            valid = True
            for pi in range(num_obj[i]):
                # if the object is out of the boundary, then, we can skip this action
                center = [i_prev_poses[pi * 2] + IMAGE_PAD_WIDTH, i_prev_poses[pi * 2 + 1] + IMAGE_PAD_WIDTH]
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
            if valid:
                for pi in range(num_obj[i]):
                    # # if the object is out of the boundary, then, we can skip this action
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
                    mask = i_rotated_mask_objs[pi]
                    points = np.argwhere(mask == 255)
                    points = np.expand_dims(points, axis=0)
                    M = cv2.getRotationMatrix2D(
                        (
                            i_prev_poses[pi * 2] + IMAGE_PAD_WIDTH,
                            i_prev_poses[pi * 2 + 1] + IMAGE_PAD_WIDTH,
                        ),
                        -i_output[pi * 3 + 2],
                        1,
                    )
                    M[0, 2] += i_output[pi * 3]
                    M[1, 2] += i_output[pi * 3 + 1]
                    new_points = cv2.transform(points, M)
                    post_points_pad.append(list(np.transpose(points[0])))
                    post_new_points_pad.append(list(np.transpose(new_points[0])))
            validations.append(valid)
            if valid:
                for pi in range(num_obj[i]):
                    post_new_points_pad[pi] = np.clip(post_new_points_pad[pi][0], 0, IMAGE_PAD_SIZE - 1), np.clip(post_new_points_pad[pi][1], 0, IMAGE_PAD_SIZE - 1)
                    post_points_pad[pi] = np.clip(post_points_pad[pi][0], 0, IMAGE_PAD_SIZE - 1), np.clip(post_points_pad[pi][1], 0, IMAGE_PAD_SIZE - 1)
                    generated_color_image[post_new_points_pad[pi]] = color_image[
                        post_points_pad[pi]
                    ]
                    generated_depth_image[post_new_points_pad[pi]] = depth_image[
                        post_points_pad[pi]
                    ]
                    if plot:
                        cv2.circle(
                            generated_color_image,
                            (i_prev_poses[pi * 2 + 1] + 48, i_prev_poses[pi * 2] + 48),
                            3,
                            (255, 255, 255),
                            -1,
                        )
                if plot:
                    cv2.arrowedLine(
                        generated_color_image,
                        (action_start_ori[i][1] + 48, action_start_ori[i][0] + 48),
                        (action_end_ori[i][1] + 48, action_end_ori[i][0] + 48),
                        (255, 0, 255),
                        2,
                        tipLength=0.4,
                    )
                generated_color_image = utils.rotate(generated_color_image, angle=-i_rotated_angle)
                generated_depth_image = utils.rotate(generated_depth_image, angle=-i_rotated_angle)
                generated_color_image = generated_color_image[
                    IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, :
                ]
                generated_depth_image = generated_depth_image[
                    IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF, IMAGE_PAD_WIDTH:IMAGE_PAD_DIFF
                ]
                generated_color_image = cv2.medianBlur(generated_color_image, 5)
                generated_depth_image = generated_depth_image.astype(np.float32)
                generated_depth_image = cv2.medianBlur(generated_depth_image, 5)

            generated_color_images.append(generated_color_image)
            generated_depth_images.append(generated_depth_image)

        return generated_color_images, generated_depth_images, validations


def _get_sign_line(pose0, pose1, pose2):
    """
    Line is from pose1 to pose2.
    if value > 0, pose0 is on the left side of the line.
    if value = 0, pose0 is on the same line.
    if value < 0, pose0 is on the right side of the line.
    """
    return (pose2[0] - pose1[0]) * (pose0[1] - pose1[1]) - (pose0[0] - pose1[0]) * (
        pose2[1] - pose1[1]
    )


def _distance_to_line(pose0, pose1, pose2):
    """
    Line is from pose1 to pose2.
    """
    return abs(
        (pose2[0] - pose1[0]) * (pose1[1] - pose0[1])
        - (pose1[0] - pose0[0]) * (pose2[1] - pose1[1])
    ) / math.sqrt((pose2[0] - pose1[0]) ** 2 + (pose2[1] - pose1[1]) ** 2)


def _adjust_push_start_point(
    pose0,
    pose1,
    contour,
    distance=GRIPPER_PUSH_RADIUS_PIXEL,
    add_distance=GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
):
    """
    Give two points, find the most left and right point on the contour within a given range based on pose1->pose0.
    So the push will not collide with the contour
    pose0: the center of contour
    pose1: the point on the contour
    """
    r = math.sqrt((pose1[0] - pose0[0]) ** 2 + (pose1[1] - pose0[1]) ** 2)
    dx = round(distance / r * (pose0[1] - pose1[1]))
    dy = round(distance / r * (pose1[0] - pose0[0]))
    pose2 = (pose0[0] + dx, pose0[1] + dy)
    pose3 = (pose1[0] + dx, pose1[1] + dy)
    pose4 = (pose0[0] - dx, pose0[1] - dy)
    pose5 = (pose1[0] - dx, pose1[1] - dy)
    pose1_sign23 = _get_sign_line(pose1, pose2, pose3)
    pose1_sign45 = _get_sign_line(pose1, pose4, pose5)
    assert pose1_sign23 * pose1_sign45 < 0
    center_distance = _distance_to_line(pose1, pose2, pose4)
    max_distance = 0
    for p in range(0, len(contour)):
        test_pose = contour[p][0]
        test_pose_sign23 = _get_sign_line(test_pose, pose2, pose3)
        test_pose_sign45 = _get_sign_line(test_pose, pose4, pose5)
        # in the range, between two lines
        if pose1_sign23 * test_pose_sign23 >= 0 and pose1_sign45 * test_pose_sign45 >= 0:
            # is far enough
            test_center_distance = _distance_to_line(test_pose, pose2, pose4)
            if test_center_distance >= center_distance:
                # in the correct side
                test_edge_distance = _distance_to_line(test_pose, pose3, pose5)
                if test_edge_distance < test_center_distance:
                    if test_center_distance > max_distance:
                        max_distance = test_center_distance
    diff_distance = abs(max_distance - center_distance)
    return math.ceil(diff_distance) + add_distance


def getOrientation(pts):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return angle


def _is_close(prev_pose, this_pose):
    dis = math.sqrt((this_pose[0] - prev_pose[0]) ** 2 + (this_pose[1] - prev_pose[1]) ** 2)
    if dis < CONSECUTIVE_DISTANCE_THRESHOLD / PIXEL_SIZE:
        return True
    return False


def _close_distance(prev_pose, this_pose):
    dis = math.sqrt((this_pose[0] - prev_pose[0]) ** 2 + (this_pose[1] - prev_pose[1]) ** 2)
    return dis


def sample_actions(
    color_image,
    depth_image,
    mask_objs,
    plot=False,
    start_pose=None,
    from_color=False,
    prev_move=None,
):
    """
    Sample actions around the objects, from the boundary to the center.
    Assume there is no object in "black"
    Output the rotated image, such that the push action is from left to right
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if plot:
        plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    # if the mask is in color format
    if from_color:
        ori_mask_objs = mask_objs[0]
        ori_mask_obj_centers = mask_objs[1]
        new_mask_objs = []
        for idx, mask in enumerate(mask_objs[0]):
            center = ori_mask_obj_centers[idx]
            new_mask = np.copy(mask[0])
            new_mask = new_mask.astype(np.uint8)
            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_RGB2GRAY)
            new_mask = cv2.threshold(new_mask, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            new_mask_pad = np.zeros((IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), dtype=np.uint8)
            if center[0] - 30 < 0 or center[0] + 30 >= IMAGE_PAD_SIZE or center[1] - 30 < 0 or center[1] + 30 >= IMAGE_PAD_SIZE:
                return [], [], [], [], [], [], [], []
            new_mask_pad[
                center[0] - 30 : center[0] + 30, center[1] - 30 : center[1] + 30
            ] = new_mask
            new_mask = new_mask_pad[
                IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
                IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
            ]
            new_mask_objs.append(new_mask)
        mask_objs = new_mask_objs

    # find the contour of a single object
    points_on_contour = []
    points = []
    four_idx = []
    other_idx = []
    priority_points_on_contour = []
    priority_points = []
    center = []
    binary_objs = []
    for oi in range(len(mask_objs)):
        obj_cnt = cv2.findContours(mask_objs[oi], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        obj_cnt = imutils.grab_contours(obj_cnt)
        if len(obj_cnt) == 0:
            return [], [], [], [], [], [], [], []
        obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))[-1]  # the mask r cnn could give bad masks
        if cv2.contourArea(obj_cnt) < 10:
            return [], [], [], [], [], [], [], []
        # get center
        M = cv2.moments(obj_cnt)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        center.append([cX, cY])
        # get crop of each object
        temp = np.zeros((IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), dtype=np.uint8)
        temp[
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
        ] = mask_objs[oi]
        crop = temp[
            cY + IMAGE_PAD_WIDTH - 30 : cY + IMAGE_PAD_WIDTH + 30,
            cX + IMAGE_PAD_WIDTH - 30 : cX + IMAGE_PAD_WIDTH + 30,
        ]
        assert crop.shape[0] == 60 and crop.shape[1] == 60, crop.shape
        binary_objs.append(crop)
        if plot:
            cv2.circle(plot_image, (cX, cY), 3, (255, 255, 255), -1)
        # get pca angle
        angle = getOrientation(obj_cnt)
        # get contour points
        skip_num = len(obj_cnt) // 12  # 12 possible pushes for an object
        skip_count = 0
        diff_angle_limit_four = 0.3
        target_diff_angles = np.array([0, np.pi, np.pi / 2, 3 * np.pi / 2])
        # add the consecutive move
        if prev_move:
            prev_angle = math.atan2(
                prev_move[1][1] - prev_move[0][1], prev_move[1][0] - prev_move[0][0]
            )
            pose = (cX - math.cos(prev_angle) * 2, cY - math.sin(prev_angle) * 2)
            x = pose[0]
            y = pose[1]
            diff_x = cX - x
            diff_y = cY - y
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_on_contour = (round(x), round(y))
            diff_mul = _adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
            point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
            diff_mul = _adjust_push_start_point((cX, cY), point_on_contour, obj_cnt, add_distance=0)
            test_point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
            if _is_close(prev_move[1], test_point):
                if len(priority_points) > 0:
                    prev_dis = _close_distance(prev_move[1], priority_points[0])
                    this_dis = _close_distance(prev_move[1], test_point)
                    if this_dis < prev_dis:
                        priority_points_on_contour[0] = point_on_contour
                        priority_points[0] = point
                else:
                    priority_points_on_contour.append(point_on_contour)
                    priority_points.append(point)
        # add four directions to center of object
        four_poses = [
            (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
            (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
            (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
            (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
        ]
        for pose in four_poses:
            x = pose[0]
            y = pose[1]
            diff_x = cX - x
            diff_y = cY - y
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_on_contour = (round(x), round(y))
            diff_mul = _adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
            point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
            points_on_contour.append(point_on_contour)
            points.append(point)
            four_idx.append(len(points) - 1)
        for pi, p in enumerate(obj_cnt):
            x = p[0][0]
            y = p[0][1]
            if x == cX or y == cY:
                continue
            diff_x = cX - x
            diff_y = cY - y
            test_angle = math.atan2(diff_y, diff_x)
            should_append = False
            # avoid four directions to center of object
            if np.min(np.abs(abs(angle - test_angle) - target_diff_angles)) < diff_angle_limit_four:
                should_append = False
                skip_count = 0
            elif skip_count == skip_num:
                should_append = True
            if should_append:
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                point_on_contour = (round(x), round(y))
                diff_mul = _adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
                point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
                points_on_contour.append(point_on_contour)
                points.append(point)
                other_idx.append(len(points) - 1)
                skip_count = 0
            else:
                skip_count += 1
    # random actions, adding priority points at the end
    # temp = list(zip(points_on_contour, points))
    # random.shuffle(temp)
    # points_on_contour, points = zip(*temp)
    # points_on_contour = list(points_on_contour)
    # points = list(points)
    # points.extend(priority_points)
    # points_on_contour.extend(priority_points_on_contour)
    random.shuffle(four_idx)
    random.shuffle(other_idx)
    new_points = []
    new_points_on_contour = []
    for idx in other_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    for idx in four_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    new_points.extend(priority_points)
    new_points_on_contour.extend(priority_points_on_contour)
    points = new_points
    points_on_contour = new_points_on_contour
    priority_qualified = False

    if plot:
        # loop over the contours
        for c in cnts:
            cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

    valid_points = []
    for pi in range(len(points)):
        # out of boundary
        if (
            points[pi][0] < 5
            or points[pi][0] > IMAGE_SIZE - 5
            or points[pi][1] < 5
            or points[pi][1] > IMAGE_SIZE - 5
        ):
            qualify = False
        elif pi >= len(points) - len(priority_points):
            temp = list(points[pi])
            temp[0] = max(temp[0], 5)
            temp[0] = min(temp[0], IMAGE_SIZE - 5)
            temp[1] = max(temp[1], 5)
            temp[1] = min(temp[1], IMAGE_SIZE - 5)
            points[pi] = temp
            qualify = True
            priority_qualified = True
        # clearance
        elif (
            np.sum(
                thresh[
                    points[pi][1] - GRIPPER_GRASP_WIDTH_PIXEL // 2 : points[pi][1] + GRIPPER_GRASP_WIDTH_PIXEL // 2 + 1,
                    points[pi][0] - GRIPPER_GRASP_WIDTH_PIXEL // 2 : points[pi][0] + GRIPPER_GRASP_WIDTH_PIXEL // 2 + 1,
                ]
                > 0
            )
            == 0
        ):
            qualify = True
        else:
            qualify = False
        if qualify:
            if plot:
                diff_x = points_on_contour[pi][0] - points[pi][0]
                diff_y = points_on_contour[pi][1] - points[pi][1]
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                point_to = (
                    int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
                    int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
                )
                if pi < len(other_idx):
                    cv2.arrowedLine(
                        plot_image,
                        points[pi],
                        point_to,
                        (27, 49, 247),
                        2,
                        tipLength=0.2,
                    )
                elif pi >= len(points) - len(priority_points):
                    cv2.arrowedLine(
                        plot_image,
                        tuple(points[pi]),
                        point_to,
                        (0, 255, 0),
                        2,
                        tipLength=0.2,
                    )
                else:
                    cv2.arrowedLine(
                        plot_image,
                        points[pi],
                        point_to,
                        (255, 94, 0),
                        2,
                        tipLength=0.2,
                    )
            valid_points.append([points[pi], points_on_contour[pi]])
    if start_pose is not None:
        spose = (start_pose[1], start_pose[0])
        epose = (start_pose[3], start_pose[2])
        valid_points = [[spose, epose]]
        print(valid_points)

    if plot:
        cv2.imwrite("test.png", plot_image)

    # rotate image
    rotated_color_image = []
    rotated_depth_image = []
    rotated_mask_objs = []
    rotated_angle = []
    rotated_center = []
    rotated_action = []
    rotated_binary_objs_image = []
    before_rotated_action = []
    count = 0
    for aidx, action in enumerate(valid_points):
        # padding from 224 to 320
        # color image
        color_image_pad = np.zeros((IMAGE_PAD_SIZE, IMAGE_PAD_SIZE, 3), np.uint8)
        color_image_pad[
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
        ] = color_image
        # depth image
        depth_image_pad = np.zeros((IMAGE_PAD_SIZE, IMAGE_PAD_SIZE), np.float32)
        depth_image_pad[
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
            IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
        ] = depth_image

        # compute rotation angle
        down = (0, 1)
        current = (action[1][0] - action[0][0], action[1][1] - action[0][1])
        dot = (
            down[0] * current[0] + down[1] * current[1]
        )  # dot product between [x1, y1] and [x2, y2]
        det = down[0] * current[1] - down[1] * current[0]  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle)
        # rotate images
        rotated_color = utils.rotate(color_image_pad, angle)
        rotated_depth = utils.rotate(depth_image_pad, angle)
        # rotate cropped object
        if len(binary_objs) == 1:
            # binary_objs_image = np.expand_dims(binary_objs[0], axis=-1)
            binary_objs_image = binary_objs[0]
            rotated_binary_objs = utils.rotate(binary_objs_image, angle, True)
            rotated_binary_objs = np.expand_dims(rotated_binary_objs, axis=-1)
        else:
            binary_objs_image = np.stack(binary_objs, axis=-1)
            rotated_binary_objs = utils.rotate(binary_objs_image, angle, True)
        M = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, 1)  # rotate by center
        # rotate points
        points = np.array(center)
        points = np.concatenate((points, [action[0]]), axis=0)
        points = np.expand_dims(points, axis=0)
        points = cv2.transform(points, M)[0]
        points_center = points[: len(center)]
        # clearance check
        clearance = cv2.cvtColor(rotated_color, cv2.COLOR_RGB2GRAY)
        clearance = cv2.medianBlur(clearance, 5)
        clearance = cv2.threshold(clearance, 20, 255, cv2.THRESH_BINARY)[1]
        area = np.sum(
                clearance[
                    max(0, points[-1][1] + IMAGE_PAD_WIDTH - round(GRIPPER_GRASP_WIDTH_PIXEL / 2)) : 
                    min(IMAGE_PAD_SIZE, points[-1][1] + IMAGE_PAD_WIDTH + round(GRIPPER_GRASP_WIDTH_PIXEL / 2) + 1),
                    max(0, points[-1][0] + IMAGE_PAD_WIDTH - GRIPPER_PUSH_RADIUS_PIXEL) : 
                    min(IMAGE_PAD_SIZE, points[-1][0] + IMAGE_PAD_WIDTH + GRIPPER_PUSH_RADIUS_PIXEL + 1)
                ]
                > 0
            )
        if area > 0:
            if not (priority_qualified and aidx == len(valid_points) - 1):
                continue
        rotated_color_image.append(rotated_color)
        rotated_depth_image.append(rotated_depth)
        rotated_angle.append(angle)
        rotated_center.append(np.flip(points_center, 1))
        rotated_action.append(np.flip(points[-1]))
        rotated_binary_objs_image.append(rotated_binary_objs)
        rotated_mask_obj = []
        rotated_mask_centers = []
        if from_color:
            for idx, mask in enumerate(ori_mask_objs):
                mask_color = mask[0]
                mask_depth = mask[1]
                rotated_mask_color = utils.rotate(mask_color, angle)
                rotated_mask_depth = utils.rotate(mask_depth, angle)
                rotated_mask = (rotated_mask_color, rotated_mask_depth)
                rotated_mask_obj.append(rotated_mask)
                rotated_mask_centers.append(
                    [
                        points_center[idx][1] + IMAGE_PAD_WIDTH,
                        points_center[idx][0] + IMAGE_PAD_WIDTH,
                    ]
                )
            rotated_mask_objs.append((rotated_mask_obj, rotated_mask_centers))
        else:
            for mask in mask_objs:
                mask = np.pad(mask, IMAGE_PAD_WIDTH, "constant", constant_values=0)
                rotated_mask = utils.rotate(mask, angle, True)
                rotated_mask_obj.append(rotated_mask)
            rotated_mask_objs.append(rotated_mask_obj)
        before_rotated_action.append(np.flip(action[0]))

        # if plot:
        #     rotated_image = rotated_color.copy()
        #     rotated_image_gray = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2GRAY)
        #     rotated_image_gray = rotated_image_gray.astype(np.uint8)
        #     rotated_image_gray = cv2.medianBlur(rotated_image_gray, 5)
        #     rotated_image = cv2.threshold(rotated_image_gray, 50, 255, cv2.THRESH_BINARY)[1]
        #     rotated_image = rotated_image[
        #         IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
        #         IMAGE_PAD_WIDTH : IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH,
        #     ]
        #     for ci in range(len(points_center)):
        #         cY, cX = rotated_center[-1][ci]
        #         cv2.circle(rotated_image, (cX, cY), 3, (128), -1)
        #     y1, x1 = rotated_action[-1]
        #     cv2.arrowedLine(
        #         rotated_image,
        #         (x1, y1),
        #         (x1, y1 + int(PUSH_DISTANCE / PIXEL_SIZE)),
        #         (128),
        #         2,
        #         tipLength=0.4,
        #     )
        #     cv2.circle(rotated_image, (x1, y1), 2, (200), -1)
        #     cv2.imwrite(str(count) + "test_rotated.png", rotated_image)
        #     count += 1

    return (
        rotated_color_image,
        rotated_depth_image,
        rotated_action,
        rotated_center,
        rotated_angle,
        rotated_binary_objs_image,
        before_rotated_action,
        rotated_mask_objs,
    )


def from_color_segm(color_image, plot=False):
    """
    Use Pre-defined color to do instance segmentation and output masks in binary format.
    """
    image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
    mask_objs = []
    if plot:
        pred_mask = np.zeros((224, 224), dtype=np.uint8)
    for ci in range(4):
        mask = cv2.inRange(image, colors_lower[ci], colors_upper[ci])
        if np.sum(mask > 0):
            mask_objs.append(mask)
            if plot:
                pred_mask[mask > 0] = 255 - ci * 20
                cv2.imwrite(str(ci) + "mask.png", mask)
    if plot:
        cv2.imwrite("pred.png", pred_mask)
        print("Mask R-CNN: %d objects detected" % len(mask_objs))
    return mask_objs


@torch.no_grad()
def from_maskrcnn(model, color_image, device, plot=False):
    """
    Use Mask R-CNN to do instance segmentation and output masks in binary format.
    """
    model.eval()

    image = color_image.copy()
    image = TF.to_tensor(image)
    prediction = model([image.to(device)])[0]

    mask_objs = []
    blue_idx = -1
    if plot:
        pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for idx, mask in enumerate(prediction["masks"]):
        # TODO, 0.9 can be tuned
        if prediction["scores"][idx] > 0.98:
            img = mask[0].mul(255).byte().cpu().numpy()
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            if np.sum(img == 255) < 100:
                continue
            color_mask = cv2.bitwise_and(color_image, color_image, mask=img)
            temp = cv2.cvtColor(color_mask, cv2.COLOR_RGB2HSV)
            temp = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            if np.sum(temp == 255) >= 50:
                blue_idx = idx
            mask_objs.append(img)
            if plot:
                pred_mask[img > 0] = 255 - idx * 50
                cv2.imwrite(str(idx) + "mask.png", img)
    if plot:
        cv2.imwrite("pred.png", pred_mask)
    print("Mask R-CNN: %d objects detected" % len(mask_objs), prediction["scores"].cpu())
    if blue_idx != -1 and blue_idx != 0:
        temp = mask_objs[0]
        mask_objs[0] = mask_objs[blue_idx]
        mask_objs[blue_idx] = temp
    return mask_objs


if __name__ == "__main__":
    color_image = cv2.imread(
        "logs_grasp/real_test_log/mcts/test16/data/color-heightmaps/000001.0.color.png"
    )
    # color_image = cv2.imread("tree_plot/root.0-73_140_74_103.1-72_138_108_130.2-99_132_136_132.png")
    # color_image_after = cv2.imread("logs_push/final-test/data/color_heightmaps/0002507.color.png")
    # color_image = cv2.imread("logs/action_test/data/color-heightmaps/000004.0.color.png")
    # color_image = cv2.imread(
    #     "logs_push/2021-01-24-16-07-43/data/color-heightmaps/000000.0.color.png"
    # )
    # color_image = cv2.imread("logs/vpg+&pp/p104/data/color-heightmaps/000001.0.color.png")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # color_image_after = cv2.cvtColor(color_image_after, cv2.COLOR_BGR2RGB)
    depth_image = cv2.imread(
        "logs_grasp/real_test_log/mcts/test16/data/depth-heightmaps/000001.0.depth.png",
        cv2.IMREAD_UNCHANGED,
    )
    # depth_image = cv2.imread("logs/real-maskrcnn/data/depth-heightmaps/000002.0.depth.png", cv2.IMREAD_UNCHANGED)
    # depth_image = cv2.imread("logs/old/object-detection-data/data/depth-heightmaps/000001.0.depth.png", cv2.IMREAD_UNCHANGED)
    # depth_image = cv2.imread(
    #     "logs_grasp/mcts-2021-03-21-00-31-13/data/depth-heightmaps/000019.0.depth.png",
    #     cv2.IMREAD_UNCHANGED,
    # )
    # depth_image = cv2.imread("logs/vpg+&pp/p104/data/depth-heightmaps/000001.0.depth.png", cv2.IMREAD_UNCHANGED)
    depth_image = depth_image.astype(np.float32) / 100000

    # with open('logs_push/final-test/data/actions/0002502.action.txt', 'r') as file:
    #     filedata = file.read()
    #     x, y = filedata.split(' ')
    # start_pose = [x, y]

    # cv2.imwrite('predicttruth.png', color_image_after)

    # check diff of color image and depth image
    # gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.medianBlur(gray, 5)
    # gray = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # depth_image[depth_image <= DEPTH_MIN] = 0
    # depth_image[depth_image > DEPTH_MIN] = 255
    # # depth_image = depth_image.astype(np.uint8)
    # cv2.imshow('color', gray)
    # cv2.imwrite('blackwhite', gray)
    # diff = depth_image - gray
    # diff[diff < 0] = 128
    # cv2.imshow('diff', diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    predictor = Predictor("logs_push/push_prediction_model-75.pth")
    # trainer = Trainer(
    #     "reinforcement",
    #     0,
    #     0,
    #     True,
    #     True,
    #     "logs_grasp/power1.5graspnew/models/snapshot-post-020000.reinforcement.pth",
    #     False,
    # )
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load("logs_image/maskrcnn.pth"))
    model = model.to(device)

    mask_objs = from_maskrcnn(model, color_image, device, True)
    (
        rotated_color_image,
        rotated_depth_image,
        rotated_action,
        rotated_center,
        rotated_angle,
        rotated_binary_objs,
        before_rotated_action,
        rotated_mask_objs,
    ) = sample_actions(color_image, depth_image, mask_objs, True)
    input('wait')

    generated_color_images, generated_depth_images, validations = predictor.forward(
        rotated_color_image,
        rotated_depth_image,
        rotated_action,
        rotated_center,
        rotated_angle,
        rotated_binary_objs,
        rotated_mask_objs,
        True,
    )
    for idx, img in enumerate(generated_color_images):
        overlay = color_image
        # added_image = cv2.addWeighted(generated_color_images[idx], 0.8, overlay, 0.4, 0)
        added_image = generated_color_images[idx].copy()
        img = cv2.cvtColor(added_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(idx) + "predict.png", img)
        img = generated_depth_images[idx]
        img[img <= DEPTH_MIN] = 0
        img[img > DEPTH_MIN] = 255
        cv2.imwrite(str(idx) + "predictgray.png", img)

    # generated_color_images.append(color_image)
    # generated_depth_images.append(depth_image)
    # for idx, img in enumerate(generated_color_images):

    #     if idx + 1 == len(generated_color_images) or validations[idx]:
    #         _, grasp_predictions = trainer.forward(
    #             generated_color_images[idx], generated_depth_images[idx], is_volatile=True
    #         )
    #         grasp_predictions = trainer.focus_on_target(
    #             generated_color_images[idx], grasp_predictions
    #         )
    #         best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
    #         predicted_value = np.max(grasp_predictions)
    #         grasp_pred_vis = trainer.get_prediction_vis(
    #             grasp_predictions, generated_color_images[idx], best_pix_ind
    #         )
    #         cv2.imwrite(str(idx) + "visualization.grasp.png", grasp_pred_vis)
    #         predicted_values = np.sum(np.sort(grasp_predictions.flatten())[:])
    #         print(idx, predicted_value, predicted_values)
    #     else:
    #         print("invalid")
    # _, grasp_predictions = trainer.forward(
    #     color_image, depth_image, is_volatile=True
    # )
    # grasp_predictions = trainer.focus_on_target(
    #     color_image, depth_image, grasp_predictions, TARGET_LOWER, TARGET_UPPER
    # )
    # best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
    # predicted_value = np.max(grasp_predictions)
    # grasp_pred_vis = trainer.get_prediction_vis(
    #     grasp_predictions, color_image, best_pix_ind
    # )
    # cv2.imwrite("visualization.grasp.png", grasp_pred_vis)
    # predicted_values = np.sum(np.sort(grasp_predictions.flatten())[:])
    # print(predicted_value, predicted_values)