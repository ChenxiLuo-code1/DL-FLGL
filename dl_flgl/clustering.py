"""FCM and manuscript-normalized, cluster-mass-weighted label impurity."""
import numpy as np

EPS = 1e-12

def memberships(x, centers, m):
    distance = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    zero = distance <= EPS
    logits = -np.log(np.maximum(distance, EPS)) / (m - 1)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    exact = zero.any(axis=1)
    weights[exact] = zero[exact] / zero[exact].sum(axis=1, keepdims=True)
    return weights

def fcm(x, clusters, m, rng, max_iter=100, tol=1e-6):
    centers = x[rng.choice(len(x), clusters, replace=False)].copy()
    for _ in range(max_iter):
        u = memberships(x, centers, m)
        um = u ** m
        updated = um.T @ x / np.maximum(um.sum(axis=0)[:, None], EPS)
        change = np.max(np.abs(updated - centers))
        centers = updated
        if change <= tol:
            break
    return memberships(x, centers, m)

def impurity(u, y):
    mass = u.sum(axis=0)
    p = u.T @ y / np.maximum(mass[:, None], EPS)
    p_tilde = p / (p.sum(axis=1, keepdims=True) + EPS)
    gini = 1 - np.square(p_tilde).sum(axis=1)
    return float(np.dot(mass / np.maximum(mass.sum(), EPS), gini))

def rule_geometry(x, u, m, h):
    um = u ** m
    mass = np.maximum(um.sum(axis=0)[:, None], EPS)
    centers = um.T @ x / mass
    variance = (um[:, :, None] * (x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=0) / mass
    return centers, h ** 2 * variance  # delta squared; epsilon added during inference
