"""Adapted from evaluation_out.py; see docs/manuscript_alignment.md for conventions."""
import numpy as np

def evaluate(y, scores, threshold=0.5):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 2 or y.shape != scores.shape or min(y.shape) == 0:
        raise ValueError("Expected equal nonempty [N,L] matrices")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid labels or scores")
    pred = np.clip(scores, 0, 1) >= threshold
    aps, oes, rls, cvs = [], [], [], []
    for truth, score in zip(y, scores):
        positive = truth == 1
        count = int(positive.sum())
        if count:
            order = np.argsort(-score, kind='stable')
            ranked = positive[order]
            aps.append(float((np.cumsum(ranked) / np.arange(1, len(truth) + 1))[ranked].mean()))
            oes.append(float(not positive[np.argmax(score)]))
            cvs.append(float((np.flatnonzero(ranked)[-1]) / len(truth)))
        if 0 < count < len(truth):
            rls.append(float((score[positive, None] <= score[None, ~positive]).mean()))
    mean = lambda v: float(np.mean(v)) if v else None
    return {'AP': mean(aps), 'HL': float(np.mean(pred != y)), 'OE': mean(oes), 'RL': mean(rls), 'CV': mean(cvs)}
