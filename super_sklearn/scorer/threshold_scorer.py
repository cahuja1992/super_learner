from super_sklearn.scorer.scorer import Scorer, sklearn, np
class ThresholdScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = sklearn.utils.multiclass.type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if y_type == "binary":
            y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, sample_weight=sample_weight, **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)