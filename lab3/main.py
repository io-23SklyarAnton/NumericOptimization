from autograd import numpy as np
from matplotlib import pyplot as plt

from lab2 import grad_descent as simple_grad_descent
from lab3 import grad_descent as impulse_grad_descent


def f(x: float | np.ndarray):
    return (np.power(x - 1, 6)) + (0.5 * np.square(x - 1))


ALPHAS = [0.02, 0.02, 0.02, 0.005, 0.02, 0.08, 0.02, 0.02]
BETAS = [0.10, 0.50, 0.70, 0.50, 0.50, 0.50, 0.50, 0.50]
START_XS = [-0.5, -0.5, -0.5, -0.5, -0.5, 0.8, -1.0, 2.0]
EPOCHS = [7, 7, 7, 7, 7, 7, 7, 7]

for alpha, start_x, epochs, beta in zip(ALPHAS, START_XS, EPOCHS, BETAS):
    fig, ax = plt.subplots(figsize=(6, 4))

    x_steps_gd = simple_grad_descent.fit_model(
        function=f,
        alpha=alpha,
        x=start_x,
        epochs=epochs,
    )
    x_steps_mom = impulse_grad_descent.fit_model(
        function=f,
        alpha=alpha,
        x=start_x,
        epochs=epochs,
        beta=beta,
    )

    y_steps_gd = f(x_steps_gd)
    y_steps_mom = f(x_steps_mom)

    x_grid = np.linspace(-1, 3, 400)
    y_grid = f(x_grid)
    ax.plot(x_grid, y_grid)

    ax.plot(x_steps_gd, y_steps_gd, "o--")

    ax.plot(x_steps_mom, y_steps_mom, "s--")

    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
