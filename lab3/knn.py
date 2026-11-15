"""Binary kNN from scratch: exhaustive squared Euclidean distances and voting.

Feature scaling is supplied by the caller. Equal distances use the saved
training row order, which must remain unchanged when comparing k values.
"""
from __future__ import annotations

import numpy as np


class BruteForceKNN:
    def __init__(self, features, labels):
        self.features = np.asarray(features, dtype=np.float64).copy()
        y = np.asarray(labels)
        if (self.features.ndim != 2 or self.features.shape[1] == 0 or len(self.features) < 5
                or y.shape != (len(self.features),) or not np.isfinite(self.features).all()
                or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2):
            raise ValueError("Need at least five finite feature rows and both binary labels 0/1")
        self.labels = y.astype(np.uint8)

    def kneighbors(self, features, n_neighbors=6, batch_size=512, progress=None):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.features.shape[1] or not np.isfinite(x).all():
            raise ValueError("Query rows must be finite and match training dimensionality")
        if (not isinstance(n_neighbors, (int, np.integer)) or not 1 <= n_neighbors <= len(self.features)
                or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("Invalid neighbour count or batch size")
        indices = np.empty((len(x), n_neighbors), dtype=np.int32)
        squared_distances = np.empty((len(x), n_neighbors), dtype=np.float64)
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            distances = np.zeros((stop-start, len(self.features)), dtype=np.float64)
            # Direct floating-point differences avoid unsigned overflow and
            # cancellation/batch dependence in the norm/dot-product identity.
            for band in range(self.features.shape[1]):
                delta = x[start:stop, band, None] - self.features[None, :, band]
                np.square(delta, out=delta)
                distances += delta
            if not np.isfinite(distances).all():
                raise FloatingPointError("Nonfinite squared Euclidean distances")
            # Columns start in saved training-row order. Stable sorting gives
            # the exact order (squared distance, training row index).
            order = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbors]
            indices[start:stop] = order
            squared_distances[start:stop] = np.take_along_axis(distances, order, axis=1)
            if progress is not None:
                progress(stop, len(x))
        return squared_distances, indices

    def vote(self, neighbor_indices, ks=(1, 3, 5)):
        indices = np.asarray(neighbor_indices)
        ks = tuple(ks)
        if (indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(self.features))
                or not ks or len(set(ks)) != len(ks)
                or any(not isinstance(k, (int, np.integer)) or k <= 0 or k % 2 == 0
                       or k > indices.shape[1] for k in ks)):
            raise ValueError("Need valid ordered neighbour indices and unique positive odd k values")
        urban_votes = np.cumsum(self.labels[indices], axis=1)
        return {k: (urban_votes[:, k-1] > k//2).astype(np.uint8) for k in ks}

    def predict(self, features, ks=(1, 3, 5), batch_size=512):
        ks = tuple(ks)
        if (not ks or any(not isinstance(k, (int, np.integer)) or k <= 0 or k % 2 == 0
                          or k > len(self.features) for k in ks)):
            raise ValueError("k must be positive, odd and no larger than training count")
        _, indices = self.kneighbors(features, n_neighbors=max(ks), batch_size=batch_size)
        return self.vote(indices, ks)
