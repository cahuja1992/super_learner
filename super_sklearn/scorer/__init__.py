import sklearn
from super_sklearn.scorer.predict_scorer import PredictScorer
from super_sklearn.scorer.probability_scorer import ProbabilityScorer
from super_sklearn.scorer.threshold_scorer import ThresholdScorer
import copy
from sklearn import metrics
from functools import partial
from super_sklearn import REGRESSION_TASKS
from super_sklearn.utils import clean_array

def make_scorer(name, score_func, optimum=1, greater_is_better=True,
                needs_proba=False, needs_threshold=False, **kwargs):
    sign = 1 if greater_is_better else -1
    if needs_proba:
        cls = ProbabilityScorer
    elif needs_threshold:
        cls = ThresholdScorer
    else:
        cls = PredictScorer
    return cls(name, score_func, optimum, sign, kwargs)

r2 = make_scorer('r2', metrics.r2_score)
mean_squared_error = make_scorer('mean_squared_error', metrics.mean_squared_error, optimum=0, greater_is_better=False)
mean_absolute_error = make_scorer('mean_absolute_error', metrics.mean_absolute_error, optimum=0, greater_is_better=False)
median_absolute_error = make_scorer('median_absolute_error', metrics.median_absolute_error, optimum=0, greater_is_better=False)

accuracy = make_scorer('accuracy', metrics.accuracy_score)
f1 = make_scorer('f1', metrics.f1_score)

roc_auc = make_scorer('roc_auc', metrics.roc_auc_score, greater_is_better=True, needs_threshold=True)
average_precision = make_scorer('average_precision', metrics.average_precision_score, needs_threshold=True)
precision = make_scorer('precision', metrics.precision_score)
recall = make_scorer('recall', metrics.recall_score)

log_loss = make_scorer('log_loss', metrics.log_loss, optimum=0, greater_is_better=False, needs_proba=True)

REGRESSION_METRICS = {}
CLASSIFICATION_METRICS = {}

for scorer in [r2, mean_squared_error, mean_absolute_error, median_absolute_error]:
    REGRESSION_METRICS[scorer.name] = scorer

for scorer in [accuracy, roc_auc, average_precision, log_loss]:
    CLASSIFICATION_METRICS[scorer.name] = scorer

for name, metric in [('precision', sklearn.metrics.precision_score),
                     ('recall', sklearn.metrics.recall_score),
                     ('f1', sklearn.metrics.f1_score)]:
    globals()[name] = make_scorer(name, metric)
    CLASSIFICATION_METRICS[name] = globals()[name]
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        globals()[qualified_name] = make_scorer(qualified_name,
                                                partial(metric,
                                                        pos_label=None,
                                                        average=average))
        CLASSIFICATION_METRICS[qualified_name] = globals()[qualified_name]

def calculate_score(solution, prediction, task_type, metric, all_scoring_functions=False):
    if all_scoring_functions:
        score = dict()
        if task_type in REGRESSION_TASKS:
            cprediction = clean_array(prediction)
            metric_dict = copy.copy(REGRESSION_METRICS)
            metric_dict[metric.name] = metric
            for metric_ in REGRESSION_METRICS:
                func = REGRESSION_METRICS[metric_]
                score[func.name] = func(solution, cprediction)

        else:
            metric_dict = copy.copy(CLASSIFICATION_METRICS)
            metric_dict[metric.name] = metric
            for metric_ in metric_dict:
                func = CLASSIFICATION_METRICS[metric_]

                try:
                    score[func.name] = func(solution, prediction)
                except ValueError as e:
                    if e.args[0] == 'multiclass format is not supported':
                        continue
                    elif e.args[0] == 'Sample-based precision, recall, ' \
                                      'fscore is not meaningful outside ' \
                                      'multilabel classification. See the ' \
                                      'accuracy_score instead.':
                        continue
                    elif e.args[0] == "Target is multiclass but " \
                                      "average='binary'. Please choose another " \
                                      "average setting.":
                        continue
                    else:
                        raise e

    else:
        if task_type in REGRESSION_TASKS:
            cprediction = clean_array(prediction)
            score = metric(solution, cprediction)
        else:
            score = metric(solution, prediction)

    return score