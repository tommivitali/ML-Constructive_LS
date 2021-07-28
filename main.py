import numpy as np
import time
from halo import Halo
from tsplib_reader import ReadTSPlib
from random_reader import RandomInstancesGenerator
from ML_greedy import MLGreedy
from candidate_list import CandidateList
from machine_learning_models import MLAdd
from utils import create_tour_from_X, compute_difference_tour_length
from drawer import plot_points_sol_intermediate


cl_method = CandidateList.Method.NearestNeighbour
ml_model = MLAdd.MLModel.SVM

print(f'Candidate List Method: {cl_method}')
print(f'ML Model: {ml_model}')
print()
print(f'--------------------------------------------------')
print()

spinner = Halo(text='Loading', spinner='dots')
format_string = "{:<15}{:<20}{:<20}{:<12}{:<12}"
header = ["Problem", "Delta ML-G", "Delta 2OPT", "Time ML-G", "Time 2OPT"]
print(format_string.format(*header))

reader = ReadTSPlib()
# reader = RandomInstancesGenerator()
for instance in reader.instances_generator():
    n_points, positions, distance_matrix, name, optimal_tour = instance

    spinner.start()

    X, X_intermediate, X_improved, time_mlg, time_2opt= MLGreedy.run(n_points, positions, distance_matrix, optimal_tour,
                                                                     cl_method=cl_method, ml_model=ml_model)

    mlg_tour = create_tour_from_X(X)
    opt_tour = np.append(optimal_tour, optimal_tour[0])

    plot_points_sol_intermediate(positions, X, X_intermediate)
    plot_points_sol_intermediate(positions, X_improved, X_intermediate)
    delta = compute_difference_tour_length(opt_tour, mlg_tour, distance_matrix)
    delta_improved = compute_difference_tour_length(opt_tour, create_tour_from_X(X_improved), distance_matrix)

    spinner.stop()

    row = [name, f'{delta * 100:.3f} %', f'{delta_improved * 100:.3f} %', f'{time_mlg:.3f} sec', f'{time_2opt:.3f} sec']
    print(format_string.format(*row))
    time.sleep(1)
