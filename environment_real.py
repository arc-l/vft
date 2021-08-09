import atexit
import socket
import math
import collections
import struct
import time
import cv2
import numpy as np

import utils
from real_camera.camera import Camera
from constants import PIXEL_SIZE, WORKSPACE_LIMITS, background_threshold


class EnvironmentReal:
    def __init__(self):
        # Connect to robot client
        self.bounds = WORKSPACE_LIMITS
        self.pixel_size = PIXEL_SIZE
        self.tcp_host_ip = "172.19.97.157"
        self.tcp_port = 30002
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.tcp_socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket_gripper.connect((self.tcp_host_ip, 63352))
        self.rtc_host_ip = "172.19.97.157"
        self.rtc_port = 30003
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))
        atexit.register(self.tcp_socket.close)
        atexit.register(self.tcp_socket_gripper.close)
        atexit.register(self.rtc_socket.close)

        # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Connect as real-time client to parse state data

        # Default home joint configuration
        self.home_joint_config = [12.58, -127.32, 127.42, -90.1, -89.83, -81.49]
        self.home_joint_config = [j / 180 * np.pi for j in self.home_joint_config]

        # Default joint speed configuration
        self.joint_acc = 8  # Safe: 1.4
        self.joint_vel = 3  # Safe: 1.05
        # self.joint_acc = 0.5  # Safe: 1.4
        # self.joint_vel = 0.5  # Safe: 1.05

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 1.2  # Safe: 0.5
        self.tool_vel = 0.3  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # Move robot to home pose
        self.go_home()
        self.setup_gripper()
        self.close_gripper()

        # Fetch RGB-D data from RealSense camera
        
        self.camera = Camera()

    def get_camera_data(self):
        # Get color and depth image from ROS service
        color_img, depth_img = self.camera.get_data()

        # cv2.imshow("color", cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("test.png", cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Remove background
        img = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
        bg_mask = cv2.inRange(img, background_threshold["low"], background_threshold["high"])
        color_img = cv2.bitwise_and(color_img, color_img, mask=bg_mask)

        return color_img, depth_img

    def parse_tcp_state_data(self, state_data, subpackage):

        # Read package header
        robot_message_type = 20
        while robot_message_type != 16:
            state_data = self.get_state()
            data_bytes = bytearray()
            data_bytes.extend(state_data)
            data_length = struct.unpack("!i", data_bytes[0:4])[0]
            # print("package length", data_length)
            robot_message_type = data_bytes[4]
        assert(robot_message_type == 16)

        byte_idx = 5
        # Parse sub-packages
        subpackage_types = {'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx + 4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            # print(package_idx)
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0, 0, 0, 0, 0, 0]
            target_joint_positions = [0, 0, 0, 0, 0, 0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 8):(byte_idx + 16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            return tool_analog_input2

        def parse_force_mode_data(data_bytes, byte_idx):
            forces = [0, 0, 0, 0, 0, 0]
            for force_idx in range(6):
                forces[force_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                byte_idx += 8
            return forces

        parse_functions = {
            'joint_data': parse_joint_data,
            'cartesian_info': parse_cartesian_info,
            'tool_data': parse_tool_data,
            'force_mode_data': parse_force_mode_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        # data_length = struct.unpack("!i", data_bytes[0:4])[0]
        # print("RTC")
        # print(struct.unpack("!i", data_bytes[0:4])[0])
        # print(struct.unpack("!i", data_bytes[4:8])[0])
        # assert(data_length == 812)
        # byte_idx = 4 + 8 + 8 * 48 + 24 + 120

        tcp_pose = [0 for i in range(6)]
        tcp_force = [0 for i in range(6)]
        byte_idx = 444
        for joint_idx in range(6):
            tcp_pose[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            byte_idx += 8
        byte_idx = 540
        for joint_idx in range(6):
            tcp_force[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            byte_idx += 8

        return tcp_pose, tcp_force

    def reset(self):

        # TODO flip box, reallocate objects

        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0, 0.0]
        tool_rotation_angle = -np.pi / 4
        tool_orientation = np.asarray(
            [
                grasp_orientation[0] * np.cos(tool_rotation_angle) - grasp_orientation[1] * np.sin(tool_rotation_angle),
                grasp_orientation[0] * np.sin(tool_rotation_angle) + grasp_orientation[1] * np.cos(tool_rotation_angle),
                0.0]) * np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation / tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi / 4, 0, 0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * \
            np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Move to box grabbing position
        box_grab_position = [0.5, -0.35, -0.12]
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2] + 0.1,
                                                                                tilted_tool_orientation[0],
                                                                                tilted_tool_orientation[1],
                                                                                tilted_tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        # self.tcp_socket.close()

        # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all(
                    [np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

        # Move to box release position
        box_release_position = [0.5, 0.08, -0.12]
        home_position = [0.49, 0.11, 0.03]
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],
                                                                                box_release_position[1],
                                                                                box_release_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.1,
                                                                                self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],
                                                                                box_release_position[1],
                                                                                box_release_position[2] + 0.3,
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.02,
                                                                                self.joint_vel * 0.02)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0] - 0.05,
                                                                                box_grab_position[1] + 0.1,
                                                                                box_grab_position[2] + 0.3,
                                                                                tilted_tool_orientation[0],
                                                                                tilted_tool_orientation[1],
                                                                                tilted_tool_orientation[2],
                                                                                self.joint_acc * 0.5,
                                                                                self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0] - 0.05,
                                                                                box_grab_position[1] + 0.1,
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.5,
                                                                                self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.1,
                                                                                self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0] + 0.05,
                                                                                box_grab_position[1],
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.1,
                                                                                self.joint_vel * 0.1)
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2] + 0.1,
                                                                                tilted_tool_orientation[0],
                                                                                tilted_tool_orientation[1],
                                                                                tilted_tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],
                                                                                home_position[1],
                                                                                home_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        # self.tcp_socket.close()

        # Block until robot reaches home position
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all(
                    [np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

    def close_gripper(self, asynch=False, pos=None):
        if pos:
            tcp_command = "SET POS " + pos + "\n"
        else:
            tcp_command = "SET POS 255\n"
        self.tcp_socket_gripper.close()
        self.tcp_socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket_gripper.connect((self.tcp_host_ip, 63352))
        self.tcp_socket_gripper.send(str.encode(tcp_command))
        self.tcp_socket_gripper.recv(8)

        max_gripper_pos = self.get_gripper_pos()
        counter = 0
        while True:
            time.sleep(0.005)
            next_gripper_pos = self.get_gripper_pos()
            if next_gripper_pos <= max_gripper_pos:
                counter += 1
            else:
                counter = 0
            if counter > 20:
                break
            max_gripper_pos = max([next_gripper_pos, max_gripper_pos])

    def open_gripper(self, asynch=False, pos=None):
        if pos:
            tcp_command = "SET POS " + pos + "\n"
        else:
            tcp_command = "SET POS 0\n"
        self.tcp_socket_gripper.close()
        self.tcp_socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket_gripper.connect((self.tcp_host_ip, 63352))
        self.setup_gripper()
        self.tcp_socket_gripper.send(str.encode(tcp_command))
        self.tcp_socket_gripper.recv(8)

        prev_gripper_pos = self.get_gripper_pos()
        while True:
            time.sleep(0.2)
            next_gripper_pos = self.get_gripper_pos()
            if next_gripper_pos >= prev_gripper_pos:
                break
            prev_gripper_pos = next_gripper_pos

        # if not asynch:
        #     time.sleep(1.5)

    def get_gripper_pos(self):
        tcp_command = "GET POS\n"
        self.tcp_socket_gripper.send(str.encode(tcp_command))
        info = self.tcp_socket_gripper.recv(8).decode("utf-8").split()
        current_pose = int(info[-1])
        return current_pose

    def setup_gripper(self):
        self.tcp_socket_gripper.send(str.encode("SET FOR 100\n"))
        self.tcp_socket_gripper.recv(8)
        self.tcp_socket_gripper.send(str.encode("SET SPE 100\n"))
        self.tcp_socket_gripper.recv(8)

    def get_state(self):
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        state_data = self.tcp_socket.recv(2048)
        # self.tcp_socket.close()
        return state_data

    def move_to(self, tool_position, tool_orientation, speed_scale=1.0):
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],
                                                                            tool_position[1],
                                                                            tool_position[2],
                                                                            tool_orientation[0],
                                                                            tool_orientation[1],
                                                                            tool_orientation[2],
                                                                            self.tool_acc * speed_scale,
                                                                            self.tool_vel * speed_scale)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches target tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        self.tcp_socket.settimeout(0.5)
        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) <
                        self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
            # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)]
            # + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]),
            # np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2))
            # for j in range(3,6)])
            try:
                tcp_state_data = self.tcp_socket.recv(2048)
                prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            # except socket.timeout:
            except:
                print("TCP socket Timeout!!!!")
                self.tcp_socket.close()
                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                self.tcp_socket.settimeout(0.5)
                self.tcp_socket_gripper.close()
                self.tcp_socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket_gripper.connect((self.tcp_host_ip, 63352))
                self.rtc_socket.close()
                self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

            time.sleep(0.01)
        time.sleep(0.2)
        # self.tcp_socket.close()

    def protected_move_to(self, tool_position, tool_orientation, speed_scale=1.0, force_max=50):
        tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],
                                                                           tool_position[1],
                                                                           tool_position[2],
                                                                           tool_orientation[0],
                                                                           tool_orientation[1],
                                                                           tool_orientation[2],
                                                                           self.tool_acc * speed_scale,
                                                                           self.tool_vel * speed_scale)
        self.tcp_socket.send(str.encode(tcp_command))
        # self.rtc_socket.settimeout(1)
        self.rtc_socket.close()
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))
        pose, force = self.parse_rtc_state_data(self.rtc_socket.recv(1108))
        self.rtc_socket.settimeout(0.5)
        pose_history = collections.deque([pose], 100)
        start_time = time.time()
        max_force = 0
        print('Protected move to...')
        # try:
        while not all([np.abs(pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            try:
                pose, force = self.parse_rtc_state_data(self.rtc_socket.recv(1108))
            except socket.timeout:
                print("RTC socket Timeout!!!!")
                self.rtc_socket.close()
                self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))
                self.rtc_socket.settimeout(0.5)
            if np.linalg.norm(np.asarray(force[0: 3])) > force_max:
                print("Collision Found!!!!", np.linalg.norm(np.asarray(force[0: 3])))
                self.move_to(pose_history[0], [tool_orientation[0],
                                               tool_orientation[1], tool_orientation[2]], speed_scale)
                return False
            else:
                max_force = max(max_force, np.linalg.norm(np.asarray(force[0: 3])))
            pose_history.append(pose)
            # time.sleep(0.05)
            time.sleep(0.001)
            if time.time() - start_time > 6:
                print('TIMEOUT!!!!!!!!')
                return False
        # except Exception as ex:
        #     prnit('?????????????????????????????')
        #     traceback.print_exception(type(ex), ex, ex.__traceback__)
        # while not all([np.abs(pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
        #     pose, force = self.parse_rtc_state_data(self.rtc_socket.recv(1108))
        #     if np.linalg.norm(np.asarray(force[0: 3])) > force_max:
        #         print("Collision Found!!!!")
        #         self.move_to(pose_history[0], [tool_orientation[0],
        #                                        tool_orientation[1], tool_orientation[2]], speed_scale)
        #         return False
        #     pose_history.append(pose)
        #     # time.sleep(0.05)
        #     time.sleep(0.001)
        #     if time.time() - start_time > 5:
        #         print('TIMEOUT!!!!!!!!')
        #         return False
        time.sleep(0.2)
        print("Max Force during this move", max_force)

        return True

    def move_joints(self, joint_configuration):

        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(2048)
        state_data = self.tcp_socket.recv(2048)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j])
                       < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(2048)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)

        # self.tcp_socket.close()

    def go_home(self):

        # self.move_to([-0.2, -0.18, 0.4], [0, -3.14, 0])
        self.joint_acc = 1.4  # Safe: 1.4
        self.joint_vel = 1.05  # Safe: 1.05
        self.move_joints(self.home_joint_config)
        self.joint_acc = 8  # Safe: 1.4
        self.joint_vel = 3  # Safe: 1.05

    # Note: must be preceded by close_gripper()

    def check_grasp(self):
        return self.get_gripper_pos() < 220
        # return True

        # state_data = self.get_state()
        # tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        # return tool_analog_input2 > 0.26

    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle):
        print('Executing: grasp at (%f, %f, %f, %f)' % (position[0], position[1], position[2], heightmap_rotation_angle))

        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0, 0.0]
        if heightmap_rotation_angle > np.pi / 2 and heightmap_rotation_angle < np.pi * 3 / 2:
            heightmap_rotation_angle = heightmap_rotation_angle - np.pi
        elif heightmap_rotation_angle >= np.pi * 3 / 2 and heightmap_rotation_angle <= np.pi * 2:
            heightmap_rotation_angle = heightmap_rotation_angle - np.pi * 2
        heightmap_rotation_angle = -heightmap_rotation_angle + np.pi / 2
        tool_rotation_angle = heightmap_rotation_angle / 2
        tool_orientation = np.asarray(
            [
                grasp_orientation[0] * np.cos(tool_rotation_angle) -
                grasp_orientation[1] * np.sin(tool_rotation_angle),
                grasp_orientation[0] * np.sin(tool_rotation_angle) +
                grasp_orientation[1] * np.cos(tool_rotation_angle),
                0.0]) * np.pi

        # Attempt grasp
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, WORKSPACE_LIMITS[2][0] + 0.02)
        self.open_gripper()
        self.move_to([position[0], position[1], position[2] + 0.1],
                        [tool_orientation[0], tool_orientation[1], 0.0], 0.5)
        move_success = self.protected_move_to([position[0], position[1], position[2]], [
            tool_orientation[0], tool_orientation[1], 0.0], 0.1)
        grasp_success = False
        if move_success:
            self.close_gripper()
            # bin_position = [-0.294, -0.465, 0.4]
            bin_position = [-0.32, 0, 0.5]
            # If gripper is open, drop object in bin and check if grasp is successful

            self.move_to([position[0], position[1], position[2] + 0.1],
                            [tool_orientation[0], tool_orientation[1], 0.0], 0.5)
            self.close_gripper()

            grasp_success = self.check_grasp()
            # grasp_success = int(input("Successfully grasped? "))

            if grasp_success:
                self.move_to([position[0], position[1], bin_position[2]], [
                    tool_orientation[0], tool_orientation[1], 0.0])
                self.move_to([bin_position[0], bin_position[1], bin_position[2]], [
                    tool_orientation[0], tool_orientation[1], 0.0])
                self.close_gripper()
                grasp_success = self.check_grasp()
                self.open_gripper()
        else:
            self.move_to([position[0], position[1], position[2] + 0.1],
                            [tool_orientation[0], tool_orientation[1], 0.0], 0.5)
        self.go_home()
        return grasp_success

    def push(self, pose0, pose1):
        print(f"Executing: push from {pose0} to {pose1}")

        # Compute tool orientation from heightmap rotation angle
        push_orientation = [1.0, 0.0]
        tool_rotation_angle = math.atan2(pose1[1] - pose0[1], pose1[0] - pose0[0])
        if tool_rotation_angle > np.pi / 2 and tool_rotation_angle < np.pi * 3 / 2:
            tool_rotation_angle = tool_rotation_angle - np.pi
        elif tool_rotation_angle >= np.pi * 3 / 2 and tool_rotation_angle <= np.pi * 2:
            tool_rotation_angle = tool_rotation_angle - np.pi * 2
        tool_rotation_angle = tool_rotation_angle + np.pi / 2
        tool_rotation_angle = tool_rotation_angle / 2
        tool_orientation = np.asarray(
            [
                push_orientation[0] * np.cos(tool_rotation_angle) -
                push_orientation[1] * np.sin(tool_rotation_angle),
                push_orientation[0] * np.sin(tool_rotation_angle) +
                push_orientation[1] * np.cos(tool_rotation_angle),
                0.0]) * np.pi

        # Compute push direction and endpoint (push to right of rotated heightmap)
        push_endpoint = np.asarray(pose1)

        # Push only within workspace limits
        position = np.asarray(pose0).copy()
        position[0] = min(max(position[0], WORKSPACE_LIMITS[0][0]), WORKSPACE_LIMITS[0][1])
        position[1] = min(max(position[1], WORKSPACE_LIMITS[1][0]), WORKSPACE_LIMITS[1][1])
        position[2] = max(position[2] + 0.02, WORKSPACE_LIMITS[2][0] + 0.02)  # Add buffer to surface
        push_startpoint = position

        self.close_gripper()
        self.move_to([push_startpoint[0],
                        push_startpoint[1],
                        push_startpoint[2] + 0.1],
                        [tool_orientation[0],
                        tool_orientation[1],
                        tool_orientation[2]],
                        1.0)
        down_success = self.protected_move_to([push_startpoint[0],
                                                push_startpoint[1],
                                                push_startpoint[2]],
                                                [tool_orientation[0],
                                                tool_orientation[1],
                                                tool_orientation[2]],
                                                0.1)
        if down_success:
            print("Pushing...")
            self.protected_move_to([push_endpoint[0],
                                    push_endpoint[1],
                                    push_startpoint[2]],
                                    [tool_orientation[0],
                                    tool_orientation[1],
                                    tool_orientation[2]],
                                    0.1, 80)
        self.move_to([push_endpoint[0],
                        push_endpoint[1],
                        push_startpoint[2] + 0.1],
                        [tool_orientation[0],
                            tool_orientation[1],
                            tool_orientation[2]],
                        1.0)

        self.go_home()

        push_success = True
        time.sleep(0.1)

        return push_success