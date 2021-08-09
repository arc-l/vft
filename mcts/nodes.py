"""Node for MCTS"""

import math
from collections import defaultdict
import numpy as np
from constants import PIXEL_SIZE, MCTS_DISCOUNT, MCTS_DISCOUNT_CONS, MCTS_TOP, GRASP_Q_PUSH_THRESHOLD
from mcts.push import is_consecutive
from memory_profiler import profile
import gc
import psutil


class PushSearchNode:
    """MCTS search node for push prediction."""

    def __init__(self, state, prev_move=None, parent=None):
        self.state = state
        self.prev_move = prev_move
        self.parent = parent
        self.children = []
        self._number_of_visits = 0.0
        self._results = []
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_actions().copy()
        return self._untried_actions

    @property
    def q(self):
        return self._results

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        untried_actions = self.untried_actions
        expanded = False
        child_node = self
        while len(untried_actions) > 0:
            action = untried_actions.pop()
            next_state = self.state.move(action)
            if next_state is None:
                self.state.remove_action(action)
            else:
                child_node = PushSearchNode(next_state, action, parent=self)
                self.children.append(child_node)
                expanded = True
                break

        return expanded, child_node

    def is_terminal_node(self):
        return self.state.is_push_over()

    def rollout(self):
        current_rollout_state = self.state
        discount_accum = 1
        results = []
        discounts = []
        while not current_rollout_state.is_push_over():
            possible_moves = current_rollout_state.get_actions()
            if len(possible_moves) == 0:
                break
            action = self.rollout_policy(possible_moves)
            new_rollout_state = current_rollout_state.move(action)
            if new_rollout_state is None:
                if current_rollout_state == self.state:
                    untried_actions = self.untried_actions
                    untried_actions.remove(action)
                current_rollout_state.remove_action(action)
                break
            if is_consecutive(new_rollout_state, current_rollout_state):
                if new_rollout_state.level <= 2:
                    discount_accum *= 1
                else:
                    discount_accum *= MCTS_DISCOUNT_CONS
            else:
                discount_accum *= MCTS_DISCOUNT
            current_rollout_state = new_rollout_state  
            results.append(current_rollout_state.push_result)
            discounts.append(discount_accum)      

        if len(results) > 0:
            result_idx = np.argmax(results)
            return results[result_idx] * discounts[result_idx], results[result_idx]
        else:
            return current_rollout_state.push_result * discount_accum, current_rollout_state.push_result

    def backpropagate(self, result, high_q):
        self._number_of_visits += 1.0
        if high_q <= self.state.push_result:
            high_q = self.state.push_result
            result = high_q
        self._results.append(result)
        if self.parent:
            if is_consecutive(self.parent.state, self.state):
                if self.state.level <= 2:
                    discount_factor = 1
                else:
                    discount_factor = MCTS_DISCOUNT_CONS
            else:
                discount_factor = MCTS_DISCOUNT
            self.parent.backpropagate(result * discount_factor, high_q)

    def best_child(self, c_param=np.sqrt(2), top=MCTS_TOP):
        choices_weights = [
            (sum(sorted(c.q)[-top:]) / min(c.n, top)) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def best_child_top(self, top=MCTS_TOP):
        # didn't complete a benchmark with this part, but is should be used >>>>>
        # directly_grasp = [c.state.push_result for c in self.children]
        # if max(directly_grasp) > GRASP_Q_PUSH_THRESHOLD:
        #     return self.children[np.argmax(directly_grasp)]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        choices_weights = [
            (sum(sorted(c.q)[-top:]) / min(c.n, top)) for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0