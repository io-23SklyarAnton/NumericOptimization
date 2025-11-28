from autograd import grad
from autograd import numpy as np


def grad_descent_step(
        function: callable,
        alpha: float,
        current_x: float,
) -> float:
    function_grad = grad(function)
    new_x = current_x - (alpha * function_grad(current_x))

    return new_x


def fit_model(
        function: callable,
        epochs: int,
        alpha: float,
        x: float,
) -> list[float]:
    x_steps = [x]
    for _ in range(epochs):
        x = grad_descent_step(
            function=function,
            alpha=alpha,
            current_x=x
        )
        x_steps.append(x)

    return np.array(x_steps)