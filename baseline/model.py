from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.csr import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


class NbSVC(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        alpha: float = 1.,
        **hparams
    ) -> None:
        self.hparams = hparams
        self.alpha = alpha
        self.Model = SVC
        self.hparams["probability"] = True  # nbsvm requires prob prediction

    @staticmethod
    def _softmax(logits):
        return np.exp(logits) / np.expand_dims(np.exp(logits).sum(-1), -1)

    @staticmethod
    def _inv_sigmoid(prob):
        return -np.log((1 / prob[:, 1]) - 1.)

    def _pr(self, x, y_i, y):
        p = self.alpha + x[y == y_i].sum(0)
        p_norm = p / ((y == y_i).sum() + self.alpha)
        return p_norm

    def get_params(self, deep=True):
        return {
            "alpha": self.alpha,
            **self.hparams
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x, y) -> None:
        self.classes_ = unique_labels(y)
        self._r = [None for _ in range(len(self.classes_))]
        self.clfs = [
            self.Model(**self.hparams)
            for _ in range(len(self.classes_))
        ]

        for i in self.classes_:
            y_train = np.array([1. if yi == i else 0. for yi in y])
            x, y_train = check_X_y(x, y_train, accept_sparse=True)

            self._r[i] = sparse.csr_matrix(
                np.log(self._pr(x, 1, y_train) / self._pr(x, 0, y_train)))

            x_nb = x.multiply(self._r[i])
            self.clfs[i].fit(x_nb, y_train)

    def predict_proba(self, x: csr_matrix) -> np.ndarray:
        logits = np.zeros((x.shape[0], len(self.classes_)))
        for i in self.classes_:
            x_nb: csr_matrix = sparse.csr_matrix(x).multiply(self._r[i])
            model_prob = self.clfs[i].predict_proba(x_nb)
            logit = self._inv_sigmoid(model_prob)
            logits[:, i] = logit
        probs = self._softmax(logits)
        return probs

    def predict(self, x: csr_matrix) -> np.ndarray:
        probs = self.predict_proba(x)
        return probs.argmax(-1)


class NbLogisticRegression(NbSVC):

    def __init__(
        self,
        alpha: float = 1.,
        **hparams
    ) -> None:
        self.hparams = hparams
        self.alpha = alpha

        self.Model = LogisticRegression
