__all__ = [
    "fit_model"
]

from autograd import grad
from autograd import numpy as np


def __grad_descent_step(
        function: callable,
        alpha: float,
        beta: float,
        current_x: float,
        p: float,
) -> tuple[float, float]:
    function_grad = grad(function)
    new_p = beta * p + function_grad(current_x)

    new_x = current_x - (alpha * new_p)

    return new_x, new_p


def fit_model(
        function: callable,
        epochs: int,
        alpha: float,
        beta: float,
        x: float,
) -> np.ndarray:
    x_steps = [x]
    p = 0.0
    for _ in range(epochs):
        x, p = __grad_descent_step(
            function=function,
            alpha=alpha,
            current_x=x,
            beta=beta,
            p=p
        )
        x_steps.append(x)

    return np.array(x_steps)
