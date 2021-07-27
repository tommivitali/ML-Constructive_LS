from os import listdir
from os.path import isfile, join, exists
from hide_output import HideOutput
from instance_solver import Solver
import numpy as np


class ReadTSPlib:

    def __init__(self):
        self.path = './data/TSPlib/'
        self.files = [f for f in listdir(f'{self.path}instances/') if isfile(join(f'{self.path}instances/', f))]
        self.files = ['kroA100.tsp', 'kroC100.tsp', 'rd100.tsp', 'eil101.tsp', 'lin105.tsp', 'pr107.tsp', 'pr124.tsp',
                      'bier127.tsp', 'ch130.tsp', 'pr136.tsp', 'gr137.tsp', 'pr144.tsp', 'kroA150.tsp', 'pr152.tsp',
                      'u159.tsp', 'rat195.tsp', 'd198.tsp', 'kroA200.tsp', 'gr202.tsp', 'ts225.tsp', 'tsp225.tsp',
                      'pr226.tsp', 'gr229.tsp', 'gil262.tsp', 'pr264.tsp', 'a280.tsp', 'pr299.tsp', 'lin318.tsp',
                      'rd400.tsp', 'fl417.tsp', 'gr431.tsp', 'pr439.tsp', 'pcb442.tsp', 'd493.tsp', 'att532.tsp',
                      'u574.tsp', 'rat575.tsp', 'd657.tsp', 'gr666.tsp', 'u724.tsp', 'rat783.tsp', 'pr1002.tsp',
                      'u1060.tsp', 'vm1084.tsp', 'pcb1173.tsp', 'd1291.tsp', 'rl1304.tsp', 'rl1323.tsp', 'nrw1379.tsp',
                      'fl1400.tsp', 'u1432.tsp', 'fl1577.tsp', 'd1655.tsp', 'vm1748.tsp']

        self.distance_formula_dict = {
            'EUC_2D': self.distance_euc,
            'ATT': self.distance_att,
            'GEO': self.distance_geo
        }

    def instances_generator(self):
        for file in self.files:
            yield self.read_instance(join(f'{self.path}instances/', file))

    def read_instance(self, filename):
        # read raw data
        with open(filename) as file_object:
            data = file_object.read()
        lines = data.splitlines()

        # get current instance information
        name = lines[0].split(' ')[1]
        n_points = np.int(lines[3].split(' ')[1])
        distance = lines[4].split(' ')[1]
        distance_formula = self.distance_formula_dict[distance]

        # read all data points for the current instance
        positions = np.zeros((n_points, 2))
        for i in range(n_points):
            line_i = lines[6 + i].split(' ')
            positions[i, 0] = float(line_i[1])
            positions[i, 1] = float(line_i[2])

        distance_matrix = ReadTSPlib.create_dist_matrix(n_points, positions, distance_formula)
        optimal_tour = self.get_optimal_solution(name, positions)

        return n_points, positions, distance_matrix, name, optimal_tour

    def get_optimal_solution(self, name, positions):
        filename = f'{self.path}optimal/{name}.npy'
        if exists(filename):
            optimal_tour = ReadTSPlib.load_optimal_solution(filename)
        else:
            optimal_tour = ReadTSPlib.compute_optimal_solution(positions)
        return optimal_tour

    @staticmethod
    def load_optimal_solution(filename):
        return np.load(filename)

    @staticmethod
    def compute_optimal_solution(positions):
        with HideOutput():
            _, opt = Solver.solve(positions)
        return opt

    @staticmethod
    def create_dist_matrix(nPoints, positions, distance_formula):
        distance_matrix = np.zeros((nPoints, nPoints))
        for i in range(nPoints):
            for j in range(i, nPoints):
                distance_matrix[i, j] = distance_formula(positions[i], positions[j])
        distance_matrix += distance_matrix.T
        return distance_matrix

    @staticmethod
    def distance_euc(zi, zj):
        delta_x = zi[0] - zj[0]
        delta_y = zi[1] - zj[1]
        return round(np.sqrt(delta_x ** 2 + delta_y ** 2), 0)

    @staticmethod
    def distance_att(zi, zj):
        delta_x = zi[0] - zj[0]
        delta_y = zi[1] - zj[1]
        rij = np.sqrt((delta_x ** 2 + delta_y ** 2) / 10.0)
        tij = float(rij)
        if tij < rij:
            dij = tij + 1
        else:
            dij = tij
        return dij

    @staticmethod
    def distance_geo(zi, zj):
        RRR = 6378.388
        lat_i, lon_i = ReadTSPlib.get_lat_long(zi)
        lat_j, lon_j = ReadTSPlib.get_lat_long(zj)
        q1 = np.cos(lon_i - lon_j)
        q2 = np.cos(lat_i - lat_j)
        q3 = np.cos(lat_i + lat_j)
        return float(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    @staticmethod
    def get_lat_long(z):
        lat = ReadTSPlib.to_radiant(z[0])
        lon = ReadTSPlib.to_radiant(z[1])
        return lat, lon

    @staticmethod
    def to_radiant(angle):
        _deg = float(angle)
        _min = angle - _deg
        return np.pi * (_deg + 5.0 * _min / 3.0) / 180.0
