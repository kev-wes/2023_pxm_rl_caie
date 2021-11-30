# RL-PxM
This repo implements a mixed integer linear program and a genetic algorithm according to the following publications:
1.	Ladj, A., Benbouzid-Si Tayeb, F., Varnier, C.: An integrated prognostic based hybrid genetic-immune algorithm for scheduling jobs and predictive maintenance. In: 2016 IEEE Congress on Evolutionary Computation (CEC), pp. 2083–2089. IEEE, Piscataway, NJ (2016). doi: 10.1109/CEC.2016.7744045
2.	Ladj, A., Varnier, C., Tayeb, F.B.-S., Zerhouni, N.: Exact and heuristic algorithms for post prognostic decision in a single multifunctional machine. International Journal of Prognostics and Health Management, vol. 8 (2017)

The repo is split into 5 parts: 1 Initialization, 2 Mixed Linear Integer Programming, 3 Genetic Algorithm, 4 Reinforcement Learning, 5 Evaluation and Visualization. 
#### 1 Initialization
Initializes Variables, e.g., problem size, cost factors, jobs and deterioration rates.
#### 2 Mixed Linear Integer Programming
Solves small problem instances (<20) using an exact solver. Warning: The algorithm is currently bugged because it does not discount the trailing job bin as zero cost, which messes up the algorithm.
#### 3 Genetic Algorithm
Implements a GA that solves bigger instances.
#### 4 Reinforcement Learning
Empty stub that should implement an RL algorithm. I decided to do this on Python within a whole new repo.
#### 5 Evaluation
Final evaluation and comparison of algorithms. Currently only MILP and GA are implemented, but the script is generic and can be extended to any algorithm that can deliver its result in a specified format. Implements cost functions and a chart.

### Dependencies
1 Initialization: none
2 Mixed Linear Integer Programming: 1 Initialization
3 Genetic Algorithm: 1 Initialization
4 Reinforcement Learning: none
5 Evaluation: 1 Initialization, 2 Mixed Linear Integer Programming, 3 Genetic Algorithm
