from super_sklearn.scorer.scorer import Scorer, sklearn, np
class PredictScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        type_true = sklearn.utils.multiclass.type_of_target(y_true)
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or \
                type_true == 'continuous':
            pass
        elif type_true in ['binary', 'multiclass']:
            y_pred = np.argmax(y_pred, axis=1)
        elif type_true == 'multilabel-indicator':
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        else:
            raise ValueError(type_true)

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, sample_weight=sample_weight, **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)