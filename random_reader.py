import random
import scipy.spatial.distance
import numpy as np
from hide_output import HideOutput
from instance_solver import Solver


class RandomInstancesGenerator:

    def __init__(self, n_points=(100, 1000), n_instances=20, seed=0):
        self.n_points_min = n_points[0]
        self.n_points_max = n_points[1]
        self.n_instances = n_instances

        random.seed(seed)

    def instances_generator(self):
        for n_instance in range(self.n_instances):
            yield self.generate_instance(n_instance)

    def generate_instance(self, n_instance):
        np.random.seed(n_instance)

        # extract number of points
        n_points = random.randint(self.n_points_min, self.n_points_max)
        # generate data points
        positions = np.random.uniform(-0.5, 0.5, size=n_points * 2).reshape((n_points, 2))
        # compute distance matrix
        distance_matrix = RandomInstancesGenerator.distance_mat(positions)
        # compute optimal tour
        optimal_tour = RandomInstancesGenerator.compute_optimal_solution(positions)

        return n_points, positions, distance_matrix, f'rand{n_points}', optimal_tour

    @staticmethod
    def create_upper_matrix(values, size):
        upper = np.zeros((size, size))
        r = np.arange(size)
        mask = r[:, None] < r
        upper[mask] = values
        return upper

    @staticmethod
    def distance_mat(pos):
        distance = RandomInstancesGenerator.create_upper_matrix(scipy.spatial.distance.pdist(pos, "euclidean"), pos.shape[0])
        distance = np.round((distance.T + distance) * 10000, 0) / 10000
        return distance

    @staticmethod
    def compute_optimal_solution(positions):
        with HideOutput():
            _, opt = Solver.solve(positions)
        return opt
