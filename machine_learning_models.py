import numpy as np
import pickle
from enum import Enum
from pickle_load import renamed_loads
from sklearn.preprocessing import MinMaxScaler


class MLAdd:

    class MLModel(Enum):
        def __str__(self):
            strings = {
                1: "Baseline",
                2: "Nearest Neighbour",
                3: "Linear",
                4: "SVM",
                5: "Ensemble",
                6: "Linear Underbalanced",
                8: "Optimal Tour",
                9: "Always True",
                10: "Always False"
            }
            return strings.get(self.value, "Invalid ML Model")

        Baseline = 1
        NearestNeighbour = 2
        Linear = 3
        SVM = 4
        Ensemble = 5
        LinearUnderbalance = 6
        OptimalTour = 8
        AllTrue = 9
        AllFalse = 10

    def __init__(self, model):
        if not isinstance(model, MLAdd.MLModel):
            raise TypeError(f'model argument should be MLModel, given {type(model)}')
        self.model = model

        if self.model == MLAdd.MLModel.Baseline:
            with open('ML/models/baseline/model_baseline_first.pickle', 'rb') as handle:
                self.model1 = renamed_loads(handle.read())
            with open('ML/models/baseline/model_baseline_second.pickle', 'rb') as handle:
                self.model2 = renamed_loads(handle.read())
        if self.model == MLAdd.MLModel.Linear:
            with open('ML/models/linear/model_linear_first.pickle', 'rb') as handle:
                self.model1 = pickle.load(handle)
            with open('ML/models/linear/scaler_linear_first.pickle', 'rb') as handle:
                self.scaler1 = pickle.load(handle)
            with open('ML/models/linear/model_linear_second.pickle', 'rb') as handle:
                self.model2 = pickle.load(handle)
            with open('ML/models/linear/scaler_linear_second.pickle', 'rb') as handle:
                self.scaler2 = pickle.load(handle)
        if self.model == MLAdd.MLModel.LinearUnderbalance:
            with open('ML/models/linear_underb/model_linear_underb_first.pickle', 'rb') as handle:
                self.model1 = pickle.load(handle)
            with open('ML/models/linear_underb/scaler_linear_underb_first.pickle', 'rb') as handle:
                self.scaler1 = pickle.load(handle)
            with open('ML/models/linear/model_linear_second.pickle', 'rb') as handle:
                self.model2 = pickle.load(handle)
            with open('ML/models/linear/scaler_linear_second.pickle', 'rb') as handle:
                self.scaler2 = pickle.load(handle)
        if self.model == MLAdd.MLModel.SVM:
            with open('ML/new_SVM/new_svm_first.pickle', 'rb') as handle:
                self.model1 = renamed_loads(handle.read())
            with open('ML/new_SVM/new_svm_sc_second2.pickle', 'rb') as handle:
                self.model2 = renamed_loads(handle.read())
            with open('ML/new_SVM/new_scaler_second2.pickle', 'rb') as handle:
                self.scaler2 = renamed_loads(handle.read())
        if self.model == MLAdd.MLModel.Ensemble:
            with open('ML/new_MAJ/ensemble_maj_first-4.pickle', 'rb') as handle:
                self.model1 = renamed_loads(handle.read())
            with open('ML/new_MAJ/ensemble_maj_second-4.pickle', 'rb') as handle:
                self.model2 = renamed_loads(handle.read())

    def __call__(self, distance, distance_vector, solution_vector, in_opt):
        if self.model == MLAdd.MLModel.AllTrue:
            return True
        if self.model == MLAdd.MLModel.AllFalse:
            return False
        if self.model == MLAdd.MLModel.OptimalTour:
            return in_opt
        if self.model == MLAdd.MLModel.NearestNeighbour:
            return distance == 1
        if self.model == MLAdd.MLModel.Baseline:
            if distance == 1:
                return self.model1.predict(distance_vector.reshape(1, -1))
            if distance == 2:
                return self.model2.predict(distance_vector.reshape(1, -1))
            return False
        if self.model == MLAdd.MLModel.SVM:
            if distance == 1:
                X = np.concatenate((distance_vector, solution_vector)).reshape(1, -1)
                if abs(self.model1.decision_function(X)) < 100:
                    return False
                return self.model1.predict(X)
            if distance == 2:
                X = np.concatenate((distance_vector, solution_vector)).reshape(1, -1)
                if abs(self.model2.decision_function(X)) < 375:
                    return False
                return self.model2.predict(self.scaler2.transform(X))
            return False
        if self.model == MLAdd.MLModel.Linear or self.model == MLAdd.MLModel.LinearUnderbalance:
            if distance == 1:
                X = self.scaler1.transform(distance_vector.reshape(1, -1))
                if abs(self.model1.decision_function(X)) < 4000:
                    return False
                return self.model1.predict(X)
            if distance == 2:
                X = self.scaler2.transform(distance_vector.reshape(1, -1))
                if abs(self.model2.decision_function(X)) < 4000:
                    return False
                return self.model2.predict(X)
            return False
        if self.model == MLAdd.MLModel.Ensemble:
            if distance == 1:
                X = np.concatenate((distance_vector, solution_vector)).reshape(1, -1)
                return self.model1.predict(X)
            if distance == 2:
                X = np.concatenate((distance_vector, solution_vector)).reshape(1, -1)
                return self.model2.predict(X)
            return False
        raise ValueError(f'{self.model} not implemented yet')
