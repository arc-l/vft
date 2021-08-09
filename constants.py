import numpy as np
import math

IS_REAL = False

if IS_REAL:
    WORKSPACE_LIMITS = np.asarray([[-0.227, 0.221], [-0.676, -0.228], [0.18, 0.4]])
else:
    WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])

# image
PIXEL_SIZE = 0.002
IMAGE_SIZE = 224
IMAGE_OBJ_CROP_SIZE = 60  # this is related to the IMAGE_SIZE and PIXEL_SIZE
IMAGE_PAD_SIZE = math.ceil(IMAGE_SIZE * math.sqrt(2) / 32) * 32  # 320
IMAGE_PAD_WIDTH = math.ceil((IMAGE_PAD_SIZE - IMAGE_SIZE) / 2)  # 48
IMAGE_PAD_DIFF = IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH  # 272

# gripper
GRIPPER_GRASP_INNER_DISTANCE = 0.08
GRIPPER_GRASP_INNER_DISTANCE_PIXEL = math.ceil(GRIPPER_GRASP_INNER_DISTANCE / PIXEL_SIZE)  # 40
GRIPPER_GRASP_OUTER_DISTANCE = 0.125
GRIPPER_GRASP_OUTER_DISTANCE_PIXEL = math.ceil(GRIPPER_GRASP_OUTER_DISTANCE / PIXEL_SIZE)  # 63
GRIPPER_GRASP_WIDTH = 0.022
GRIPPER_GRASP_WIDTH_PIXEL = math.ceil(GRIPPER_GRASP_WIDTH / PIXEL_SIZE)  # 11
GRIPPER_GRASP_SAFE_WIDTH = 0.025
GRIPPER_GRASP_SAFE_WIDTH_PIXEL = math.ceil(GRIPPER_GRASP_SAFE_WIDTH / PIXEL_SIZE)  # 13
GRIPPER_PUSH_RADIUS = 0.015
GRIPPER_PUSH_RADIUS_PIXEL = math.ceil(GRIPPER_PUSH_RADIUS / PIXEL_SIZE)  # 8
GRIPPER_PUSH_RADIUS_SAFE_PIXEL = math.ceil(GRIPPER_PUSH_RADIUS_PIXEL * math.sqrt(2))  # 12
PUSH_DISTANCE = 0.05 + PIXEL_SIZE * GRIPPER_PUSH_RADIUS_SAFE_PIXEL  # 0.074
PUSH_DISTANCE_PIXEL = math.ceil(PUSH_DISTANCE / PIXEL_SIZE)  # 37

CONSECUTIVE_ANGLE_THRESHOLD = 0.2  # radius
CONSECUTIVE_DISTANCE_THRESHOLD = 0.05  # cm

if IS_REAL:
    GRASP_Q_PUSH_THRESHOLD = 1.0
    GRASP_Q_GRASP_THRESHOLD = 0.7
else:
    GRASP_Q_PUSH_THRESHOLD = 1.0
    GRASP_Q_GRASP_THRESHOLD = 0.8
MCTS_ROLLOUTS = 150
MCTS_MAX_LEVEL = 4
MCTS_DISCOUNT_CONS = 0.8
MCTS_DISCOUNT = 0.6
MCTS_TOP = 3

NUM_ROTATION = 16
DEPTH_MIN = 0.01  # depth filter, count valid object

PUSH_BUFFER = 0.05

COLOR_SPACE = (
    np.asarray(
        [
            [78, 121, 167],  # blue
            [89, 161, 79],  # green
            [156, 117, 95],  # brown
            [242, 142, 43],  # orange
            [237, 201, 72],  # yellow
            [186, 176, 172],  # gray
            [255, 87, 89],  # red
            [176, 122, 161],  # purple
            [118, 183, 178],  # cyan
            [255, 157, 167],  # pink
        ]
    )
    / 255.0
)

# norm
COLOR_MEAN = [0.0241, 0.0213, 0.0165]
COLOR_STD = [0.1122, 0.0988, 0.0819]
DEPTH_MEAN = [0.0019]
DEPTH_STD = [0.0091]
BINARY_IMAGE_MEAN = [0.0646, 0.0125]
BINARY_IMAGE_STD = [0.2410, 0.1113]
BINARY_OBJ_MEAN = [0.1900]
BINARY_OBJ_STD = [0.3707]

# pre training values
PUSH_Q = 0.25
GRASP_Q = 0.5

background_threshold = {
    "low": np.array([0, 0, 125], np.uint8),
    "high": np.array([255, 255, 255], np.uint8),
}  # white
BG_THRESHOLD = {
    "low": np.array([0, 0, 0], np.uint8),
    "high": np.array([180, 255, 50], np.uint8),
}  # black
# colors
real_purple_lower = np.array([100, 143, 0], np.uint8)
real_purple_upper = np.array([126, 255, 255], np.uint8)
# rgb(69, 108, 149) to hsv(105 137 149)
blue_lower = np.array([95, 87, 99], np.uint8)
blue_upper = np.array([115, 187, 199], np.uint8)
# rgb(79, 143, 70) to hsv(56 130 143)
green_lower = np.array([48, 80, 87], np.uint8)
green_upper = np.array([64, 180, 187], np.uint8)
# 11  97 131
brown_lower = np.array([8, 57, 91], np.uint8)
brown_upper = np.array([14, 137, 171], np.uint8)
# 15 209 206
orange_lower = np.array([12, 159, 156], np.uint8)
orange_upper = np.array([18, 255, 255], np.uint8)
# 23 177 202
yellow_lower = np.array([20, 127, 152], np.uint8)
yellow_upper = np.array([26, 227, 252], np.uint8)
# 158, 148, 146 to 5 19 158
gray_lower = np.array([0, 0, 108], np.uint8)
gray_upper = np.array([15, 56, 208], np.uint8)
# rgb(217, 74, 76) to 0 168 217
red_lower = np.array([0, 118, 172], np.uint8)
red_upper = np.array([10, 218, 255], np.uint8)
# rgb(148, 104, 136) to 158  76 148
purple_lower = np.array([148, 26, 98], np.uint8)
purple_upper = np.array([167, 126, 198], np.uint8)
# rgb(101, 156, 151) to 87  90 156
cyan_lower = np.array([77, 40, 106], np.uint8)
cyan_upper = np.array([97, 140, 206], np.uint8)
# rgb(216, 132, 141) to 177  99 216
pink_lower = np.array([168, 49, 166], np.uint8)
pink_upper = np.array([187, 149, 255], np.uint8)
colors_lower = [
    blue_lower,
    green_lower,
    brown_lower,
    orange_lower,
    yellow_lower,
    gray_lower,
    red_lower,
    purple_lower,
    cyan_lower,
    pink_lower,
]
colors_upper = [
    blue_upper,
    green_upper,
    brown_upper,
    orange_upper,
    yellow_upper,
    gray_upper,
    red_upper,
    purple_upper,
    cyan_upper,
    pink_upper,
]

if IS_REAL:
    TARGET_LOWER = real_purple_lower
    TARGET_UPPER = real_purple_upper
else:
    TARGET_LOWER = blue_lower
    TARGET_UPPER = blue_upper

# black backgroud sim
# color_mean = [0.0235, 0.0195, 0.0163]
# color_std = [0.1233, 0.0975, 0.0857]
# depth_mean = [0.0022]
# depth_std = [0.0089]

# random sim
# color_mean = [0.0272, 0.0225, 0.0184]
# color_std = [0.1337, 0.1065, 0.0922]
# depth_mean = [0.0020]
# depth_std = [0.0073]

# # binary
# binary_mean = [0.2236]
# binary_std = [0.4167]
# used_binary_mean = [0.0635, 0.0289]
# used_binary_std = [0.2439, 0.1675]


# total_obj = 5

# resolution and padding resolution
# heightmap_resolution = 0.002
# resolution = 224
# resolution_pad = math.ceil(resolution * math.sqrt(2) / 32) * 32
# padding_width = math.ceil((resolution_pad - resolution) / 2)
# resolution_crop = 60