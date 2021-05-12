import sys
from copy import deepcopy
from typing import List

import networkx as nx
import numpy as np

MIN, MAX = sys.float_info.min, sys.float_info.max

"""
In this assignment, you will apply local search to graph coloring.

Given a k-colorable graph, your goal is to color the graph with k colors so as to minimize
the number of monochromatic edges.

Since local search algorithms require enumerating through many possible neighboring
configuratoins, you should use lazy iterators when evaluating local moves :

Example:

def lazy_func(n):
    for x in range(n):
        yield x**2

for x in lazy_func(5):
    print(x)
    
output : 0, 1, 4, 9, 16
    
Note that itertools provides some useful (lazy) generators such as "combinations"

from itertools import combinations

which will allow you to iterate over all subsets of a baseset with a fixed size.
"""


class Coloring:
    """A coloring of n objects with k colors
    """

    def __init__(self, n: int, k: int, coloring=None):
        self.n = n
        self.k = k
        self.coloring = coloring or np.random.randint(0, k, size=n)

    def __getitem__(self, item):
        return self.coloring[item]

    def get_coloring(self):
        return np.array(self.coloring)


def random_k_colorable_graph(n=20, p=0.5, k=3):
    """Returns a k-colorable graph with n vertices
    """
    G = nx.Graph()
    # add nodes
    for i in range(n):
        G.add_node(i)
    real_colors = np.random.randint(0, k, size=n)

    # add edges
    for i in range(n):
        for j in range(i + 1, n):
            if real_colors[i] != real_colors[j]:
                if np.random.uniform() < p:
                    G.add_edge(i, j)
    return G


def get_local_neighbors(coloring: Coloring, radius) -> List:
    """Gets local neighbors within radius of a coloring.

    Given a current coloring C = (c1,...,cn) or V(G), this function will
    (lazily) generate all neighboring colorings C' such that the hamming
    distance is at most radius i.e. N = {C' | d(C,C') <=radius}.

    Example:
        coloring = [1,2,1,1], k = 2, radius = 1
        N = {
        [1,2,1,1],
        [2,2,1,1],
        [1,1,1,1],
        [1,2,2,1],
        [1,2,1,2],
        }

    Params:
        coloring : the current coloring of V(G)
        radius : maximum number of vertex/color pairs to change in current coloring
    """
    # (Lazily) generate all colorings within radius of the current coloring.
    # TODO

    pass


def compare(G, current, neighbor):
    """Compares the current coloring to a neighboring coloring

    Evaluates the change in the number of monochromatic edges between
    the current coloring and the neighboring coloring

    Params:
        G: the graph being colored
        current : The current coloring
        neighbor : The neighboring coloring
    """

    # determine the vertices whose colors have changed
    # compute the difference in the number of monochromatic edges between the colorings.
    # NOTE: this operation should take O(n*k) time where k is the number of vertices
    # in neighbor whose color differs from current.
    # NOTE: To be clear, this function should return a POSITIVE number when the neighbor
    # coloring has fewer monochromatic edges
    # TODO
    pass


def local_search_step(G, current_coloring : Coloring, radius):
    """Updates the current coloring by evaluating all neighbors within the given radius
    """

    # copy the current coloring in case it is modified in get_local_neighbors
    # you can remove this if it is not applicable -- you should not have to copy
    # the coloring elsewhere
    current_coloring_copy = deepcopy(current_coloring)
    best_diff, best_coloring = MIN, current_coloring_copy
    for neighbor in get_local_neighbors(current_coloring_copy, radius):
        diff = compare(G, neighbor)
        best_diff = max(best_diff, diff)
        if diff == best_diff:
            # you can copy neighbor here if necessary
            best_coloring = neighbor
    return best_coloring

def local_search(G, k, max_radius, num_steps = 100):
    """Runs local search to find a k-coloring of G

    Params:
        G: the graph
        k: the k in k-coloring
        max_radius: the radius to use in local search neighborhood
        num_steps: the number of local search steps to exectue

    Returns:
        the
    """
    #TODO: implement the local search procedure
