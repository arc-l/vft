import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--file_reward', action='store', type=str)
# parser.add_argument('--file_action', action='store', type=str)
parser.add_argument('--log', action='store', type=str)
args = parser.parse_args()

new = os.path.isfile(os.path.join(args.log, "transitions", "label-value.log.txt"))
print(args.log)

if new:
    reward_file = os.path.join(args.log, "transitions", "label-value.log.txt")
    action_file = os.path.join(args.log, "transitions", "executed-action.log.txt")

    reward_log = np.loadtxt(reward_file, delimiter=' ')
    print(f"total action is: {len(reward_log)}")
    print(f"get the target object: {np.sum(reward_log == 1)}")
    print(f"average number: {len(reward_log) / np.sum(reward_log == 1)}")

    action_log = np.loadtxt(action_file, delimiter=' ')
    assert len(reward_log) == len(action_log)
    action_log = action_log[:, 0]
    print(f"grasp success: {np.sum(reward_log[action_log == 1]) / np.sum(action_log == 1)}")
    print(reward_log[action_log == 1])
else:
    reward_file = os.path.join(args.log, "transitions", "reward-value.log.txt")
    action_file = os.path.join(args.log, "transitions", "executed-action.log.txt")

    reward_log = np.loadtxt(reward_file, delimiter=' ')
    print(f"total action is: {len(reward_log)}")
    print(f"get the target object: {np.sum(reward_log == 1)}")
    print(f"average number: {len(reward_log) / np.sum(reward_log == 1)}")

    action_log = np.loadtxt(action_file, delimiter=' ')
    assert len(reward_log) == len(action_log)
    action_log = action_log[:, 0]
    print(f"grasp success: {np.sum(reward_log[action_log == 0]) / np.sum(action_log == 0)}")
    print(reward_log[action_log == 0])