from super_sklearn.scorer.scorer import Scorer
class ProbabilityScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.
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
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, sample_weight=sample_weight, **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)