__all__ = [
    "fit_model"
]

from autograd import jacobian
from autograd import numpy as np


def __grad_descent_step(
        function_jacobian,
        alpha: float,
        beta: float,
        features: np.ndarray[float],
        p: float,
) -> tuple[np.ndarray[float], float]:
    new_p = beta * p + function_jacobian(features)

    new_x = features - (alpha * new_p)

    return new_x, new_p


def fit_model(
        function: callable,
        epochs: int,
        alpha: float,
        beta: float,
        features: tuple[float, float],
) -> np.ndarray:
    function_jacobian = jacobian(function)
    features = np.array(features, dtype=float)

    features_steps = [features.copy()]
    p = 0.0
    for _ in range(epochs):
        features, p = __grad_descent_step(
            function_jacobian=function_jacobian,
            alpha=alpha,
            features=features,
            beta=beta,
            p=p
        )
        features_steps.append(features.copy())

    return np.array(features_steps)
