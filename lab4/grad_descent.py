__all__ = [
    "fit_model"
]

from autograd import jacobian
from autograd import numpy as np


def __grad_descent_step(
        function_jacobian,
        alpha: float,
        features: np.ndarray[float],
) -> float:
    new_features = features - (alpha * function_jacobian(features))

    return new_features


def fit_model(
        function: callable,
        epochs: int,
        alpha: float,
        features: tuple[float, float],
) -> np.ndarray:
    function_jacobian = jacobian(function)
    features = np.array(features, dtype=float)

    features_steps = [features.copy()]
    for _ in range(epochs):
        features = __grad_descent_step(
            function_jacobian=function_jacobian,
            alpha=alpha,
            features=features,
        )
        features_steps.append(features.copy())

    return np.array(features_steps)
