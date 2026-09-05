"""SSFCM rule selection and additive TSK fuzzy dictionary."""
import numpy as np
from .clustering import EPS, fcm, impurity, rule_geometry

class SSFCM:
    def __init__(self, config):
        self.config = config

    def fit(self, x, y):
        c = self.config
        self.n_features_ = x.shape[1]
        self.M_ = c.M if c.M is not None else int(np.ceil(c.M_fraction * x.shape[1]))
        if not 1 <= self.M_ <= x.shape[1] or c.C > len(x):
            raise ValueError("Require 1 <= M <= D and C <= training sample count")
        rng = np.random.default_rng(c.seed)
        candidates = []
        for candidate_id in range(c.B):
            indices = rng.choice(x.shape[1], self.M_, replace=False)
            subspace = x[:, indices]
            u = fcm(subspace, c.C, c.m, rng, c.fcm_max_iter, c.fcm_tol)
            centers, widths = rule_geometry(subspace, u, c.m, c.h)
            candidates.append((impurity(u, y), candidate_id, indices, centers, widths))
        candidates.sort(key=lambda row: (row[0], row[1]))
        self.rules_ = []
        self.selected_ = []
        for score, cid, indices, centers, widths in candidates[:c.B_prime]:
            self.selected_.append({"candidate": cid, "impurity": score, "features": indices.tolist()})
            self.rules_.extend((indices, v, w) for v, w in zip(centers, widths))
        return self

    def firing_strength(self, x):
        # Log-sum-exp evaluates the stated additive average without underflow.
        logs = []
        for indices, center, width in self.rules_:
            log_mu = -(x[:, indices] - center) ** 2 / (width + EPS)
            logs.append(np.logaddexp.reduce(log_mu, axis=1) - np.log(len(indices)))
        logs = np.column_stack(logs)
        normalizer = np.logaddexp.reduce(logs, axis=1, keepdims=True)
        return np.exp(logs - normalizer)

    def transform(self, x):
        phi = self.firing_strength(x)
        extended = np.column_stack((np.ones(len(x)), x))
        return (phi[:, :, None] * extended[:, None, :]).reshape(len(x), -1)
