import numpy as np
import scipy.spatial.distance


def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    r = np.arange(size)
    mask = r[:, None] < r
    upper[mask] = values
    return upper


def create_distance_matrix(pos):
    distance = create_upper_matrix(scipy.spatial.distance.pdist(pos, "euclidean"), pos.shape[0])
    distance = np.round((distance.T + distance) * 10000, 0) / 10000
    return distance


def create_tour_from_X(X):
    tour = np.array([0, X[0, 0]])
    while tour[0] != tour[-1]:
        [a, b] = X[tour[-1]]
        x = a if a != tour[-2] else b
        tour = np.append(tour, x)
    return tour


def compute_tour_lenght(tour, distance_matrix):
    return np.sum(np.array([distance_matrix[a, b] for a, b in zip(tour, tour[1:])]))


def compute_difference_tour_length(tour_opt, tour_mlg, distance_matrix):
    mlg_tour_length = compute_tour_lenght(tour_mlg, distance_matrix)
    opt_tour_length = compute_tour_lenght(tour_opt, distance_matrix)
    return (mlg_tour_length - opt_tour_length) / opt_tour_length
