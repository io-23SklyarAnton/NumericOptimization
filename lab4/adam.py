__all__ = [
    "fit_model"
]

from autograd import jacobian
from autograd import numpy as np


def __grad_descent_step(
        function_jacobian,
        alpha: float,
        features: np.ndarray[float],
        p: np.ndarray,
        r: np.ndarray,
        p1: float,
        p2: float,
        t: int,
        epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jacobian_in_point = function_jacobian(features)
    new_p = p1 * p + (1.0 - p1) * jacobian_in_point
    p_hat = new_p / (1.0 - p1 ** t)

    new_r = p2 * r + (1.0 - p2) * (jacobian_in_point * jacobian_in_point)
    r_har = new_r / (1.0 - p2 ** t)

    new_x = features - alpha * (p_hat / (epsilon + r_har ** 0.5))

    return new_x, new_p, new_r


def fit_model(
        function: callable,
        epochs: int,
        alpha: float,
        p1: float,
        p2: float,
        epsilon: float,
        features: tuple[float, float],
) -> np.ndarray:
    function_jacobian = jacobian(function)
    features = np.array(features, dtype=float)

    features_steps = [features.copy()]

    p = np.zeros_like(features, dtype=float)
    r = np.zeros_like(features, dtype=float)
    for t in range(1, epochs + 1):
        features, p, r = __grad_descent_step(
            function_jacobian=function_jacobian,
            alpha=alpha,
            features=features,
            p=p,
            r=r,
            p1=p1,
            p2=p2,
            t=t,
            epsilon=epsilon,
        )
        features_steps.append(features.copy())

    return np.array(features_steps)
