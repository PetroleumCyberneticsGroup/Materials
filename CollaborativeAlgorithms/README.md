# Collaborative Algorithms
The paper of Angga et al. presents a collaborative algorithmic framework that is effective for solving a multi-task optimization scenario where the evaluation of their objectives consists of two parts: The first part involves a common computationally heavy function, e.g., a numerical simulation, while the second part further evaluates the objective by performing additional, significantly less computationally-intensive calculations [1].

This directory contains Python scripts, i.e., "C-GA.py", "C-PSO.py", and "C-GD.py", for solving the first problem set defined in [1] using both the collaborative and non-collaborative algorithms. In those scripts, an input argument called "coop" will decide to run either the collaborative (if set True) or the non-collaborative (if set False) algorithms.

# Execution Procedure
1. Define path of the output directory in the "C-GA.py", "C-PSO.py", or "C-GD.py" scripts, particularly in a variable called "out".
2. Execute the runner scripts, i.e., "RunnerC-GA.py", "RunnerC-PSO.py", or "RunnerC-GD.py".

# References
[1] I. G. A. G. Angga, M. Bellout, P. E. S. Bergmo, P. A. Slotte, and C. F. Berg, “Collaborative optimization by shared objective function data,” 2022, (under review)
