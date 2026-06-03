"""Small modelling helpers used when heavyweight ML libraries are unavailable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


@dataclass
class Standardiser:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_


def fit_standardiser(x: np.ndarray) -> Standardiser:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return Standardiser(mean, scale)


@dataclass
class LogisticModel:
    weights: np.ndarray
    standardiser: Standardiser

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.standardiser.transform(x)
        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        return sigmoid(np.dot(design, self.weights))


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.08,
    l2: float = 0.01,
    epochs: int = 1200,
) -> LogisticModel:
    standardiser = fit_standardiser(x)
    x_scaled = standardiser.transform(x)
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    weights = np.zeros(design.shape[1])

    for _ in range(epochs):
        probability = sigmoid(np.dot(design, weights))
        gradient = np.dot(design.T, probability - y) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient

    return LogisticModel(weights=weights, standardiser=standardiser)


@dataclass
class DecisionStump:
    feature: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(x[:, self.feature] <= self.threshold, self.left_value, self.right_value)


def _best_stump(x: np.ndarray, target: np.ndarray, features: np.ndarray) -> DecisionStump:
    best_error = np.inf
    best = DecisionStump(feature=int(features[0]), threshold=0.0, left_value=float(target.mean()), right_value=float(target.mean()))
    for feature in features:
        values = np.quantile(x[:, feature], [0.2, 0.35, 0.5, 0.65, 0.8])
        for threshold in values:
            left = x[:, feature] <= threshold
            right = ~left
            if left.sum() < 10 or right.sum() < 10:
                continue
            left_value = float(target[left].mean())
            right_value = float(target[right].mean())
            prediction = np.where(left, left_value, right_value)
            error = float(np.mean((target - prediction) ** 2))
            if error < best_error:
                best_error = error
                best = DecisionStump(int(feature), float(threshold), left_value, right_value)
    return best


@dataclass
class StumpForest:
    stumps: list[DecisionStump]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        predictions = np.column_stack([stump.predict(x) for stump in self.stumps])
        return np.clip(predictions.mean(axis=1), 0.001, 0.999)


def fit_stump_forest(x: np.ndarray, y: np.ndarray, n_estimators: int = 80, seed: int = 7) -> StumpForest:
    rng = np.random.default_rng(seed)
    stumps: list[DecisionStump] = []
    n_rows, n_features = x.shape
    feature_count = max(1, int(np.sqrt(n_features)))
    for _ in range(n_estimators):
        rows = rng.integers(0, n_rows, size=n_rows)
        features = rng.choice(n_features, size=feature_count, replace=False)
        stumps.append(_best_stump(x[rows], y[rows], features))
    return StumpForest(stumps=stumps)


@dataclass
class BoostedStumps:
    base_log_odds: float
    stumps: list[DecisionStump]
    learning_rate: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = np.full(len(x), self.base_log_odds, dtype=float)
        for stump in self.stumps:
            score += self.learning_rate * stump.predict(x)
        return sigmoid(score)


def fit_boosted_stumps(
    x: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 70,
    learning_rate: float = 0.18,
    seed: int = 7,
) -> BoostedStumps:
    rng = np.random.default_rng(seed)
    base_probability = np.clip(y.mean(), 0.001, 0.999)
    score = np.full(len(y), np.log(base_probability / (1 - base_probability)))
    stumps: list[DecisionStump] = []
    for _ in range(n_estimators):
        probability = sigmoid(score)
        residual = y - probability
        features = rng.choice(x.shape[1], size=max(1, int(np.sqrt(x.shape[1]))), replace=False)
        stump = _best_stump(x, residual, features)
        score += learning_rate * stump.predict(x)
        stumps.append(stump)
    return BoostedStumps(float(np.log(base_probability / (1 - base_probability))), stumps, learning_rate)
