# Algorithms for the b-chromatic colouring problem
A b-chromatic colouring of a graph $G$ is a proper $k$-colouring such that for every colour $c$ there exist a vertex $v$ of colour $c$ such that $v$ is adjacent to a vertex of every other colour $c'\neq c$. The b-chromatic number of $G$, denoted $\varphi(G)$, is the maximum $k$ such that $G$ admits a b-chromatic $k$-colouring. Finding a b-chromatic colouring of a general graph is NP-hard.

This package contains:
 1. Linear time implementations to find a b-chromatic colouring in trees (The code is based on the proof given in [1]) and cubic graphs,
 2. an heuristic to approximate the value of $\varphi(G)$, and
 3. an integer programming model to find a b-chromatic colouring of the total graph of $G$. 


# Bibliography
1- Manlove, D. F. (1998). Minimaximal and maximinimal optimisation problems: a partial order-based approach (Doctoral dissertation, University of Glasgow).
