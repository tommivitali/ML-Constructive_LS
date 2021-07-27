from enum import Enum, auto
from scipy.spatial import Delaunay
from pypopmusic.PyCandidatePOP import PyCandidatePOP
import numpy as np


class CandidateList:

    class Method(Enum):
        def __str__(self):
            strings = {
                1: "Nearest Neighbour",
                2: "Delaunay Triangulation",
                3: "POP Music"
            }
            return strings.get(self.value, "Invalid CL Method")

        NearestNeighbour = 1
        DelaunayTriangulation = 2
        POPMusic = 3

    @staticmethod
    def NearestNeighbour(positions, distance_matrix, length):
        n = len(positions)
        candidate_list = np.empty((n, length), dtype=int)
        for node in range(n):
            candidate_list[node] = np.argsort(distance_matrix[node])[1:length+1]
        return candidate_list

    @staticmethod
    def __triangulation(positions):
        tri = Delaunay(positions, qhull_options="QJ")
        cl = [[] for _ in range(positions.shape[0])]
        for [a, b, c] in tri.simplices:
            (b not in cl[a]) and cl[a].append(b)
            (c not in cl[a]) and cl[a].append(c)

            (a not in cl[b]) and cl[b].append(a)
            (c not in cl[b]) and cl[b].append(c)

            (a not in cl[c]) and cl[c].append(a)
            (b not in cl[c]) and cl[c].append(b)
        return cl

    @staticmethod
    def __second_degree_neighbours(cl, node):
        neighbours = cl[node]
        for first_degree_neighbour in cl[node]:
            neighbours = np.union1d(neighbours, cl[first_degree_neighbour])
        neighbours = np.delete(neighbours, np.where(neighbours == node))
        return neighbours

    @staticmethod
    def DelaunayTriangulation(positions, distance_matrix, length):
        triangulation = CandidateList.__triangulation(positions)
        candidate_list = np.empty(len(positions), dtype=object)
        for node in range(len(positions)):
            neighbours = CandidateList.__second_degree_neighbours(triangulation, node)
            distances = np.array([distance_matrix[node, n] for n in neighbours])
            indexes = np.argsort(distances)[:length]
            candidate_list[node] = neighbours[indexes]
        return candidate_list

    @staticmethod
    def POPMusic(positions, n_sampling=3):
        instance_popmusic = PyCandidatePOP(verbose=False, number_of_solutions=n_sampling)
        candidate_list, _ = instance_popmusic(positions)
        return np.array([np.array(cl) for cl in candidate_list], dtype=object)

    @staticmethod
    def compute(positions, distance_matrix, method, length=15):
        if not isinstance(method, CandidateList.Method):
            raise TypeError(f'method argument should be CandidateListMethod, given {type(method)}')

        if method == CandidateList.Method.NearestNeighbour:
            return CandidateList.NearestNeighbour(positions, distance_matrix, length)
        elif method == CandidateList.Method.DelaunayTriangulation:
            return CandidateList.DelaunayTriangulation(positions, distance_matrix, length)
        elif method == CandidateList.Method.POPMusic:
            return CandidateList.POPMusic(positions)

        raise ValueError(f'{method} not implemented yet')
