import time
import numpy as np
from utils import create_tour_from_X
from candidate_list import CandidateList
from machine_learning_models import MLAdd


class MLGreedy:

    @staticmethod
    def inner_loop(X, edge):
        [i, j] = edge
        n_nodes = 1
        n = X.shape[0]
        X[i] = np.array([j, X[i, 0]])
        X[j] = np.array([i, X[j, 0]])
        a, b = X[i, 0], i
        while n_nodes < n:
            # print(f'{n_nodes} <> {b}->{a}')
            if a == i:
                return True
            if a == -1:
                break
            if np.sum(X[a] == -1) == 2:
                break
            tmp = a
            a = X[a, 0] if X[a, 0] != b else X[a, 1]
            b = tmp
            n_nodes = n_nodes + 1
        return False

    @staticmethod
    def run(n, positions, distance_matrix, optimal_tour, cl_method=CandidateList.Method.NearestNeighbour,
            ml_model=MLAdd.MLModel.NearestNeighbour, limit=15):
        t0 = time.time()
        # create CL for each vertex
        candidate_list = CandidateList.compute(positions, distance_matrix, cl_method)
        # insert the shortest two vertices for each CL into L_P
        L_P = np.empty((2, n * 2), dtype=int)
        L_P_distances = np.empty(n * 2, dtype=int)
        for node in range(n):
            #vertices = candidate_list[node][np.argsort(distance_matrix[node, candidate_list[node]])[:2]]
            vertices = np.argsort(distance_matrix[node])[1:3]
            L_P[:, node] = [node, vertices[0]]
            L_P[:, n + node] = [node, vertices[1]]
        # sort L_P according to ascending costs c_{i,j} (before the first, then the second)
        costs_1st = np.array([distance_matrix[i, j] for [i, j] in L_P.T[:n]])
        costs_2nd = np.array([distance_matrix[i, j] for [i, j] in L_P.T[n:]])
        L_P = L_P[:, np.concatenate((np.argsort(costs_1st), np.argsort(costs_2nd) + n))]
        L_P_distances[:n] = [1] * n
        L_P_distances[n:] = [2] * n
        # initialize X
        X = np.full((n, 2), -1, dtype=int)
        # initialize ML Model
        ML_add = MLAdd(model=ml_model)
        # for l in L_P select the extreme vertices i, j of l
        for L_P_pos, [i, j] in enumerate(L_P.T):
            # if i and j have less than two connections each in X
            if np.sum(X[i] > -1) < 2 and np.sum(X[j] > -1) < 2:
                # if l do not creates a inner-loop
                if not MLGreedy.inner_loop(X.copy(), [i, j]):
                    nodes = np.concatenate(([i], candidate_list[i]))
                    nodes = np.pad(nodes, (0, limit + 1 - len(nodes)), constant_values=(-1))
                    dists = np.full(int(limit * (limit + 1) / 2), fill_value=-1, dtype=np.float)
                    edges = np.empty(int(limit * (limit + 1) / 2), dtype=object)
                    edges_in_sol = np.zeros(int(limit * (limit + 1) / 2))
                    current_pos = 0
                    for pos_node_a, node_a in enumerate(nodes[:-1]):
                        for pos_node_b, node_b in enumerate(nodes[pos_node_a + 1:]):
                            if node_a == -1 or node_b == -1:
                                edges[current_pos] = (-1, -1)
                                continue
                            dists[current_pos] = distance_matrix[node_a][node_b]
                            edges[current_pos] = (node_a, node_b)
                            edges_in_sol[current_pos] = node_b in X[node_a]
                            current_pos += 1

                    pos_i_opt = np.argwhere(optimal_tour == i)[0][0]
                    ret = (j == optimal_tour[pos_i_opt - 1] or j == optimal_tour[(pos_i_opt + 1) % len(optimal_tour)])

                    # if the ML agrees the addition of l
                    if ML_add(distance=L_P_distances[L_P_pos], distance_vector=dists, solution_vector=edges_in_sol, in_opt=ret):
                        X[i] = np.array([j, X[i, 0]])
                        X[j] = np.array([i, X[j, 0]])

        X_intermediate = np.copy(X)

        # find the hub vertex h: h = argmax_{i \in V} TD[i], where TD is the total
        # distance e.g. the sum of all the distances outgoing from a node
        TD = np.sum(distance_matrix, axis=0)
        h = np.argmin(TD)
        # select all the edges that connects free vertices and insert them into L_D
        free_vertices = np.where(np.sum(X == -1, axis=1))[0]
        free_vertices_masked = np.ma.array(free_vertices, mask=False)
        L_D = np.array([[], []], dtype=int)
        for i, vert in enumerate(free_vertices[:-1]):
            free_vertices_masked.mask[i] = True
            L_D = np.hstack((L_D, [np.full(np.ma.count(free_vertices_masked), vert),
                                   free_vertices_masked.compressed()]))
        L_D = L_D.T
        # compute the savings values wrt h for each edge in L_D,
        # where s_{i, j} = c_{i, h} + c_{h, j} - c_{i, j}
        s = np.array([distance_matrix[i, h] + distance_matrix[h, j] - distance_matrix[i, j] for [i, j] in L_D])
        # sort L_D according to the descending savings s_{i, j}
        L_D = L_D[np.argsort(-s)]
        t = 0
        # while the solution X is not complete
        while (X == -1).any():
            # select the extreme vertices i, j of l
            [i, j] = L_D[t]
            t = t + 1
            # if vertex i and vertex j have less than two connections each in X
            if np.sum(X[i] > -1) < 2 and np.sum(X[j] > -1) < 2:
                # if l do not creates a inner-loop
                if not MLGreedy.inner_loop(X.copy(), [i, j]):
                    X[i] = np.array([j, X[i, 0]])
                    X[j] = np.array([i, X[j, 0]])
        time_mlg = time.time() - t0

        X_improved, time_2opt = MLGreedy.improve_solution(X, X_intermediate, distance_matrix)
        # return X, X_intermediate, X, time_mlg, 0
        return X, X_intermediate, X_improved, time_mlg, time_2opt

    @staticmethod
    def get_free_nodes(X):
        free_nodes = np.array([i for i in range(len(X)) if (X[i] == -1).any()])
        return free_nodes

    @staticmethod
    def get_fixed_edges(X):
        fixed_edges = np.empty(np.sum(X != -1), dtype=object)
        n_fixed_e = 0
        for i, [a, b] in enumerate(X):
            if a != -1:
                fixed_edges[n_fixed_e] = (i, a)
                n_fixed_e += 1
            if b != -1:
                fixed_edges[n_fixed_e] = (i, b)
                n_fixed_e += 1
        return list(fixed_edges)

    @staticmethod
    def improve_solution(X, X_intermediate, distance_matrix):
        t0 = time.time()
        X = np.copy(X)
        fixed_edges = MLGreedy.get_fixed_edges(X_intermediate)
        free_nodes = MLGreedy.get_free_nodes(X_intermediate)
        while True:
            X_improved, improvement = MLGreedy.two_opt(X, free_nodes, fixed_edges, distance_matrix)
            if improvement == 0:
                return X_improved, time.time() - t0

    @staticmethod
    def two_opt(X, free_nodes, fixed_edges, distance_matrix):
        tour = create_tour_from_X(X)
        for i, node_ip in enumerate(tour[:-1]):
            node_in = tour[i + 1]
            if node_ip in free_nodes and node_in in free_nodes and (node_ip, node_in) not in fixed_edges:
                for j, node_jp in enumerate(tour[i+1:-1]):
                    node_jn = tour[i + j + 2]
                    if node_jp in free_nodes and node_jn in free_nodes and (node_jp, node_jn) not in fixed_edges:
                        old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                        new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                        if old_cost - new_cost > 0:
                            # print(f'({node_ip}, {node_in}) <-> ({node_jp}, {node_jn})')
                            X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                            X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                            X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                            X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                            return X, old_cost - new_cost
        return X, 0
