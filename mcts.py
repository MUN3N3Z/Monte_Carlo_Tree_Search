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
    
    def add_edge(self, action, child) -> None:
        self._edges.append(Edge(action, child))
        return

    def update_value(self, reward: float) -> None:
        self._value += reward
        return
    
    def update_visits(self) -> None:
        self._visits += 1
        return 
    
    def expand(self) -> None:
        if not self._edges:
            possible_actions = self._state.get_actions()
            children_states = [Node(self._state.successor(action)) for action in possible_actions]
            list_of_edges = [self.add_edge(action, child) for action, child in zip(possible_actions, children_states)]
            self._edges = list_of_edges
        return
    
    def next_child_to_explore(self) -> 'Edge':
        def ucb1(edge: 'Edge') -> float:
            if edge._visits == 0:
                return float('inf')
            return edge.child._value + sqrt(2 * log(self._visits) / edge._visits)
        return max(self._edges, key=ucb1)
    
    def simulate(self) -> float:
        state = self._state
        possible_actions = state.get_actions()
        while possible_actions:
            action = choice(possible_actions)
            state = state.successor(action)
            possible_actions = state.get_actions()
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

    @property
    def edges(self) -> List['Edge']:
        return self._edges
    
    def traverse(self) -> Tuple[Deque['Edge'], Node]:
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


def monte_carlo_tree_search(position: State, duration: float):
    start_time = time()
    root = RootNode(position)
    
    while time() - start_time < duration:
        # Traverse: Choode path from root to best leaf node
        edges_to_leaf_node, best_child = root.traverse()
        # Expand: Add children to best leaf node if possible
        best_child.expand()
        # Simulate: Simulate a random game from best leaf node to terminal state
        reward = best_child.simulate()
        # Backpropagate: Update value of nodes in path from root to best leaf node
        best_child.backpropagate(reward, edges_to_leaf_node)
    
    return max(root.edges, key=lambda edge: edge.child.value).action

def mcts_policy(duration: float):
    return lambda position: monte_carlo_tree_search(position, duration)