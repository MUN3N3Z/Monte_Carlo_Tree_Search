from game import State
from time import time
from deck import Card
from typing import List, Deque, Tuple, Union
from math import log, sqrt
import random
from collections import deque

class Node:
    def __init__(self, state: State):
        self._state = state
        self._edges = [] 
        self._value = 0
        self._visits = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def state(self) -> State:
        return self._state
    
    @property
    def edges(self) -> List['Edge']:
        return self._edges
    
    @property
    def visits(self) -> int:
        return self._visits
    
    def expand(self) -> Tuple[Union['Edge', None], 'Node']:
        """
            A node is expandable if it is non-terminal and has been visited. 
            Confirmed that node is non-terminal in main loop.
            Add all children nodes at once to avoid bias in exploration.
        """
        possible_actions = self.state.get_actions()
        children_nodes = [Node(self.state.successor(action)) for action in possible_actions]
        self._edges = [Edge(action, child, self) for action, child in zip(possible_actions, children_nodes)]
        random_edge = random.choice(self.edges)
        return random_edge, random_edge.child
    
    def next_child_to_explore(self, state: State) -> 'Edge':
        def ucb(edge: 'Edge') -> float:
            if edge.visits == 0:
                return float('inf')
            t = sum(e.visits for e in self.edges)
            return ((edge.child.value / edge.visits) if state.actor() == 0 else (- edge.child.value / edge.visits)) + sqrt(2 * log(t) / edge.visits)
        return max(self.edges, key=ucb) if state.actor() == 0 else min(self.edges, key=ucb)
    
    def simulate(self) -> float:
        state = self._state
        while not state.is_terminal():
            possible_actions = state.get_actions()
            random_action = random.choice(possible_actions)
            state = state.successor(random_action)
        return state.payoff()
    
    def average_payoff(self) -> float:
        return self._value / self._visits if self._visits > 0 else 0
    
    def backpropagate(self, reward: float, edges_to_root: Deque['Edge']) -> None:
        # Update visit count and reward for leaf node first
        self._visits += 1
        self._value += reward
        while edges_to_root:
            edge = edges_to_root.popleft()
            edge.update_visits()
            edge.parent._value += reward 
            edge.parent._visits += 1
        return 


class RootNode(Node):
    def __init__(self, state: State):
        super().__init__(state)
    
    def traverse(self) -> Tuple[Deque['Edge'], Node]:
        """
            Use UCB formula to guide tree traversal from root to leaf node
            Only return a node if it's a leaf node or it's expandable
        """
        node = self
        edges_to_leaf_node = deque()
        while node.edges:
            edge = node.next_child_to_explore(self.state)
            edges_to_leaf_node.appendleft(edge)
            node = edge.child
        return edges_to_leaf_node, node
    
class Edge:
    def __init__(self, action: Union[Card, List[int]], child: Node, parent: Node):
        self._action = action
        self._parent = parent
        self._child = child
        self._visits = 0
    
    @property
    def child(self) -> Node:
        return self._child
    
    @property
    def action(self) -> Union[Card, List]:
        return self._action
    
    @property
    def parent(self) -> Node:
        return self._parent
    
    @property
    def visits(self) -> int:
        return self._visits
    
    def update_visits(self) -> None:
        self._visits += 1


def monte_carlo_tree_search(state: State, duration: float):
    random.seed(19)
    start_time = time()
    root = RootNode(state)
    while (time() - start_time) < duration:
        # Traverse: Choose path from root to best leaf node
        edges_to_leaf_node, node_to_simulate_play = root.traverse()
        # Expand: Add children to best leaf node if possible
        if not node_to_simulate_play.state.is_terminal():
            new_edge, node_to_simulate_play = node_to_simulate_play.expand()
            # Expand path to leaf node
            edges_to_leaf_node.appendleft(new_edge)
        # Simulate: Simulate a random game from best leaf node to terminal state
        reward = node_to_simulate_play.simulate()
        # Backpropagate: Update value of nodes in path from root to best leaf node
        node_to_simulate_play.backpropagate(reward, edges_to_leaf_node)
    
    return max(root.edges, key=lambda edge: edge.child.average_payoff()).action if root.state.actor() == 0 else min(root.edges, key=lambda edge: edge.child.average_payoff()).action

def mcts_policy(duration: float):
    return lambda state: monte_carlo_tree_search(state, duration)