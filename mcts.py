from game import State
from time import time
from deck import Card
from typing import List, Deque, Tuple
from math import log, sqrt
from random import choice
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
    def visits(self) -> int:
        return self._visits

    def number_of_children(self) -> int:
        return len(self._edges)
    
    def add_edge(self, action, child) -> None:
        self._edges.append(Edge(action, child, self))
        return

    def update_value(self, reward: float) -> None:
        self._value += reward
        return
    
    def update_visits(self) -> None:
        self._visits += 1
        return 
    
    def add_children(self) -> None:
        possible_actions = self._state.get_actions()
        children_states = [Node(self._state.successor(action)) for action in possible_actions]
        for action, child in zip(possible_actions, children_states):
            self.add_edge(action, child)
        return
    
    def expand(self) -> 'Node':
        """
            A node is expandable if it is non-terminal and has been visited
            If a node is expandable: return a random child
            else: return same node
        """
        if not self._state.is_terminal() and self.visits > 0:
            self.add_children()
            return choice(self._edges).child
        return self
    
    def next_child_to_explore(self) -> 'Edge':
        def ucb(edge: 'Edge') -> float:
            if edge._visits == 0:
                return float('inf')
            average_child_value = edge.child.value / edge.child.number_of_children() if edge.child.number_of_children() else 0
            return average_child_value + sqrt(2 * log(self.visits) / edge.visits)
        return max(self._edges, key=ucb)
    
    def simulate(self) -> float:
        state = self._state
        while not state.is_terminal():
            possible_actions = state.get_actions()
            random_action = choice(possible_actions)
            state = state.successor(random_action)
        return state.payoff()

    def is_leaf_node(self) -> bool:
        return not self._edges
    
    def backpropagate(self, reward: float, edges_to_root: Deque['Edge']) -> None:
        while edges_to_root:
            edge = edges_to_root.popleft()
            edge.update_visits()
            edge.parent.update_value(reward)
            edge.parent.update_visits()
        return 


class RootNode(Node):
    def __init__(self, state: State):
        super().__init__(state)
        self.add_children()

    @property
    def edges(self) -> List['Edge']:
        return self._edges
    
    def traverse(self) -> Tuple[Deque['Edge'], Node]:
        """
            Use UCB formula to guide tree traversal from root to leaf node
        """
        node = self
        edges_to_leaf_node = deque()
        while not node.is_leaf_node():
            edge = node.next_child_to_explore()
            edges_to_leaf_node.appendleft(edge)
            node = edge.child
        return edges_to_leaf_node, node
    
class Edge:
    def __init__(self, action: Card | List[int], child: Node, parent: Node):
        self._action = action
        self._parent = parent
        self._child = child
        self._visits = 0
    
    def update_visits(self) -> None:
        self._visits += 1

    @property
    def child(self) -> Node:
        return self._child
    
    @property
    def action(self) -> Card | List:
        return self._action
    
    @property
    def parent(self) -> Node:
        return self._parent
    
    @property
    def visits(self) -> int:
        return self._visits


def monte_carlo_tree_search(position: State, duration: float):
    start_time = time()
    root = RootNode(position)
    
    while (time() - start_time) < duration:
        # Traverse: Choode path from root to best leaf node
        edges_to_leaf_node, best_child = root.traverse()
        # Expand: Add children to best leaf node if possible
        node_to_simulate_play = best_child.expand()
        # Simulate: Simulate a random game from best leaf node to terminal state
        reward = node_to_simulate_play.simulate()
        # Backpropagate: Update value of nodes in path from root to best leaf node
        best_child.backpropagate(reward, edges_to_leaf_node)
    
    return max(root.edges, key=lambda edge: edge.child.visits).action

def mcts_policy(duration: float):
    return lambda position: monte_carlo_tree_search(position, duration)