from game import State
from time import time
from deck import Card
from typing import List, Self
from math import log, sqrt
from random import choice

class Node:
    def __init__(self, state: State):
        self._state = state
        # List of edges from this node to its children
        self._edges = [] 
        self._value = 0
        self._visits = 0
        # Expand the node
        self.setup()
    
    @property
    def state(self) -> State:
        return self._state
    
    def setup(self) -> None:
        """
            Expands the node by adding children states and their corresponding edges
        """
        possible_actions = self._state.get_actions()
        children_states =  [Node(self._state.successor(action)) for action in possible_actions]
        list_of_edges = [Edge(action, child, self) for action, child in zip(possible_actions, children_states)]
        self.edges = list_of_edges

    def next_child_to_explore(self) -> Self:
        """
            Chooses an action (successor state) to take based on the UCB formula
        """
        # Choose arbitrarily if the node has not been visited
        if self.visits == 0:
            return self.edges[0].child
        else:
            return max(self.edges, key=lambda edge: edge.child.value +  sqrt((2 * log(self.visits) / edge.visits)) if edge.visits > 0 else 0).child # Avoid division by zero
    
    def add_edge(self, action, child) -> None:
        self.edges.append(Edge(action, child))

    def update_value(self, reward: float) -> None:
        self.value += reward
    
    def update_visits(self) -> None:
        self.visits += 1

    def explore(self):
        """
            Explores the tree from this node by selecting random actions until a terminal state is reached
        """
        pass

class Edge:
    def __init__(self, action: Card | List , child: Node, parent: Node):
        self._action = action
        self._parent = parent
        self._child = child
        self._visits = 0
    
    def update_visits(self) -> None:
        self.visits += 1

    @property
    def child(self) -> Node:
        return self.child
    
    @property
    def action(self) -> Card | List:
        return self.action
    
    @property
    def parent(self) -> Node:
        return self.parent


def monte_carlo_tree_search(position: State, duration: float):
    start_time = time()
    while time() - start_time < duration:
        node = Node(position)
        # Use UCB to select a node (traversal)
        best_child = node.next_child_to_explore()
        # If node is expandable, expand it by adding children

        # Simulate a game from the selected node by selecting actions randomly - return the reward

        # Backpropagate the result of the simulation

        # Return the action leading to child with highest value
        pass

def mcts_policy(duration: float):
    return lambda position: monte_carlo_tree_search(position, duration)
    