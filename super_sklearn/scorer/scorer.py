from abc import ABCMeta, abstractmethod
import sklearn
import numpy as np

class Scorer(object, metaclass=ABCMeta):
    def __init__(self, name, score_func, optimum, sign, kwargs):
        self.name = name
        self._kwargs = kwargs
        self._score_func = score_func
        self._optimum = optimum
        self._sign = sign

    @abstractmethod
    def __call__(self, y_true, y_pred, sample_weight=None):
        pass

    def __repr__(self):
        return self.name