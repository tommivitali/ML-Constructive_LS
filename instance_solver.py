import os
import tempfile
from pyconcorde.concorde.tsp import TSPSolver


class Solver:

    @staticmethod
    def solve(positions):
        old_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as path:
            os.chdir(path)

            solver = TSPSolver.from_data(
                positions[:, 0] * 1000,
                positions[:, 1] * 1000,
                norm="EUC_2D"
            )
            solution = solver.solve()
        os.chdir(old_dir)
        return solution.found_tour, solution.tour
