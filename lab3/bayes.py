"""Full Gaussian Bayes classifier using NumPy operations only.

Inputs share a fixed coordinate system (training-only standardization in the
lab). Covariances include the requested regularization exactly once.
"""
from __future__ import annotations

import numpy as np


class GaussianBayes:
    def __init__(self, classes, means, covariances, priors):
        self.classes = np.asarray(classes).copy()
        self.means = np.asarray(means, dtype=np.float64).copy()
        self.covariances = np.asarray(covariances, dtype=np.float64).copy()
        self.priors = np.asarray(priors, dtype=np.float64).copy()
        if self.means.ndim != 2 or len(self.means) < 2:
            raise ValueError("Expected at least two class means")
        count, dimensions = self.means.shape
        if (dimensions == 0 or self.classes.shape != (count,) or not np.all(self.classes[1:] > self.classes[:-1])
                or self.covariances.shape != (count, dimensions, dimensions)
                or self.priors.shape != (count,)):
            raise ValueError("Invalid parameter shapes or unsorted/duplicate classes")
        if (not all(np.isfinite(a).all() for a in (self.means, self.covariances, self.priors))
                or np.any(self.priors <= 0) or not np.isclose(self.priors.sum(), 1)):
            raise ValueError("Parameters must be finite; positive priors must sum to one")
        if not np.allclose(self.covariances, self.covariances.transpose(0, 2, 1), rtol=1e-12, atol=1e-14):
            raise ValueError("Covariances must be symmetric")
        self.factors = np.linalg.cholesky(self.covariances)
        self.log_determinants = 2 * np.log(np.diagonal(self.factors, axis1=1, axis2=2)).sum(axis=1)
        self.log_constants = np.log(self.priors) - .5 * (
            dimensions * np.log(2*np.pi) + self.log_determinants)

    @classmethod
    def fit(cls, features, labels, priors=None, regularization=1e-6):
        """Class means and MLE covariance, denominator N_c; no feature scaling."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError("Expected finite feature rows and one label per row")
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("Regularization must be finite and nonnegative")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Need at least two classes")
        means, covariances = [], []
        for label in classes:
            samples = x[y == label]
            if len(samples) < 2:
                raise ValueError("Need at least two samples per class")
            mean = samples.mean(axis=0)
            centered = samples - mean
            means.append(mean)
            covariances.append(centered.T @ centered / len(samples)
                               + regularization * np.eye(x.shape[1]))
        if priors is None:
            priors = np.full(len(classes), 1 / len(classes))
        return cls(classes, means, covariances, priors)

    def log_scores(self, features):
        """log p(x | c) + log P(c), via Cholesky solves; no matrix inverse."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.means.shape[1] or not np.isfinite(x).all():
            raise ValueError("Query features must be finite with the fitted dimensionality")
        scores = np.empty((len(x), len(self.classes)), dtype=np.float64)
        for column, (mean, factor) in enumerate(zip(self.means, self.factors)):
            whitened = np.linalg.solve(factor, (x-mean).T)
            quadratic = np.sum(whitened**2, axis=0)
            scores[:, column] = self.log_constants[column] - .5 * quadratic
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite Gaussian log scores")
        return scores

    def predict(self, features, batch_size=8192):
        """Return class IDs; exact equal scores choose the smaller class ID."""
        x = np.asarray(features, dtype=np.float64)
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if x.ndim != 2 or x.shape[1] != self.means.shape[1] or not np.isfinite(x).all():
            raise ValueError("Query features must be finite with the fitted dimensionality")
        labels = np.empty(len(x), dtype=self.classes.dtype)
        for start in range(0, len(x), batch_size):
            stop = start+batch_size
            scores = self.log_scores(x[start:stop])
            # Classes are sorted; NumPy argmax returns the first exact maximum.
            labels[start:stop] = self.classes[np.argmax(scores, axis=1)]
        return labels
