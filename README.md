# b-chromatic colouring
A b-chromatic colouring of a graph $G$ is a proper $k$-colouring such that for every colour $c$ there must exist a vertex $v$, with colour $c$, for which its neighbourhood is coloured with all the remaining colours. A b-chromatic $k$-colouring of $G$ is a b-chromatic colouring using $k$ colours. The b-chromatic number of $G$, denoted $\varphi(G)$, is the maximum $k$ such that $G$ admits a b-chromatic $k$-colouring. Finding a b-colouring of a general graph is NP-hard. However Irving and Manlove [1] showed that is polynomial time solvable in trees by giving implicit constructions of the b-colouring. We give linear time implementation to find a b-colouring in trees. The code is based on the proof given in [1]


# Bibliography
1- Manlove, D. F. (1998). Minimaximal and maximinimal optimisation problems: a partial order-based approach (Doctoral dissertation, University of Glasgow).
