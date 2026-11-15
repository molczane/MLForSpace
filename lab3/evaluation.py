"""Binary metrics from explicit confusion counts; no ready-made estimators."""
from __future__ import annotations

import numpy as np


def binary_confusion(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (truth.ndim != 1 or prediction.shape != truth.shape
            or not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all()):
        raise ValueError("Expected aligned 1D binary labels; exclude nodata/unlabelled first")
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (truth.astype(np.int64), prediction.astype(np.int64)), 1)
    return matrix


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all() or (matrix < 0).any()
            or not np.equal(matrix, np.floor(matrix)).all()):
        raise ValueError("Expected a 2 by 2 matrix of nonnegative integer counts")
    matrix = matrix.astype(np.int64)

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    def macro(values):
        return float(np.mean(values)) if all(value is not None for value in values) else None

    per_class = []
    for label, name in enumerate(("forest", "urban")):
        tp = int(matrix[label, label])
        support = int(matrix[label].sum())
        predicted = int(matrix[:, label].sum())
        fp, fn = predicted - tp, support - tp
        per_class.append(dict(label=label, class_name=name, support=support,
                              predicted_support=predicted, tp=tp, fp=fp, fn=fn,
                              precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                              f1=ratio(2*tp, 2*tp+fp+fn)))
    total = int(matrix.sum())
    summary = dict(evaluated_pixels=total, correct_pixels=int(np.trace(matrix)),
                   accuracy=ratio(int(np.trace(matrix)), total),
                   balanced_accuracy=macro([row["recall"] for row in per_class]),
                   macro_f1=macro([row["f1"] for row in per_class]))
    return dict(confusion=matrix.tolist(), per_class=per_class, summary=summary)
