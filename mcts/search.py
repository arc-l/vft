from tqdm import tqdm


class MonteCarloTreeSearch(object):
    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------

        """
        for i in tqdm(range(simulations_number)):
            r, v = self._tree_policy()
            if r:
                reward, high_q = v.rollout()
            else:
                reward = 0
                high_q = 0
            v.backpropagate(reward, high_q)
        # to select best child go for exploitation only
        # return self.root.best_child(c_param=0.0)
        return self.root.best_child_top(top=1)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return True, current_node
