import sklearn
import numpy as np
from super_sklearn.scorer import calculate_score
import collections

class EnsembleSklearn(sklearn.base.BaseEstimator):
    def __init__(self, ensemble_size, task_type, metric):
        self.ensemble_size = int(ensemble_size)
        self.task_type = task_type
        self.metric = metric

    def fit(self, predictions, labels, identifiers):
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []
        ensemble_size = self.ensemble_size

        n_best = 20
        indices = perf = np.zeros([predictions.shape[0]])

        for idx, prediction in enumerate(predictions):
            perf[idx] = calculate_score(labels, prediction, self.task_type, self.metric, predictions.shape[1])

        indices = np.argsort(perf)[perf.shape[0] - n_best:]

        for idx in indices:
            ensemble.append(predictions[idx])
            order.append(idx)
            ensemble_ = np.array(ensemble).mean(axis=0)
            ensemble_performance = calculate_score(
                solution=labels,
                prediction=ensemble_,
                task_type=self.task_type,
                metric=self.metric,
                all_scoring_functions=False)
            trajectory.append(ensemble_performance)
        ensemble_size -= n_best

        for _ in range(ensemble_size):
            scores = np.zeros([predictions.shape[0]])
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                scores[j] = 1 - calculate_score(
                    solution = labels,
                    prediction = ensemble_prediction,
                    task_type = self.task_type,
                    metric = self.metric,
                    all_scoring_functions = False)
                ensemble.pop()
            best = np.nanargmin(scores)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            if len(predictions) == 1:
                break

        self.indices_ = np.array(order)
        self.trajectory_ = np.array(trajectory)
        self.train_score_ = trajectory[-1]

        ensemble_members = collections.Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights
        self.identifiers_ = identifiers
        return self