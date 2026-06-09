# Algorithms for the b-chromatic colouring problem
A b-chromatic colouring of a graph $G$ is a proper $k$-colouring such that for every colour $c$ there exist a vertex $v$ of colour $c$ such that $v$ is adjacent to a vertex of every other colour $c'\neq c$. The b-chromatic number of $G$, denoted $\varphi(G)$, is the maximum $k$ such that $G$ admits a b-chromatic $k$-colouring. Finding a b-chromatic colouring of a general graph is NP-hard.

However, finding a b-chromatic colouring is polynomial-time solvable for several graph classes, e.g., trees, cubic graphs and cographs. There are also several approaches used to approximate the value of $\varphi(G)$ for a given graph $G$. This Python package contains implementations of some polynomial-time algorithms, and heuristics and integer programming models to compute and approximate $\varphi(G)$ for a given graph $G$, respectively.

This package contains:
 1. Linear time implementations to find a b-chromatic colouring in trees (The code is based on the proof given in [1]) and cubic graphs,
 2. linear time algorithm to find a b-chromatic colouring in a cubic graph,
 2. an heuristic to approximate the value of $\varphi(G)$, and
 3. an integer programming model to find a b-chromatic colouring of the total graph of $G$. 


# Bibliography
1- Manlove, D. F. (1998). Minimaximal and maximinimal optimisation problems: a partial order-based approach (Doctoral dissertation, University of Glasgow).
