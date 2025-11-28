from autograd import numpy as np
from matplotlib import pyplot as plt

from lab2.grad_descent import fit_model


def f(x: float | np.ndarray):
    return np.exp(x) - 2 * x


ALPHAS = [0.35, 0.15, 0.9]
START_XS = [-2.0, 3.0, 1.0]
EPOCHS = [6, 9, 3]

for alpha, start_x, epochs in zip(ALPHAS, START_XS, EPOCHS):
    x_steps = fit_model(
        function=f,
        alpha=alpha,
        x=start_x,
        epochs=epochs
    )
    y_steps = f(x_steps)

    features = np.linspace(-2, 3, 100)
    labels = f(features)
    plt.plot(features, labels)
    plt.plot(x_steps, y_steps, 'o--')
    plt.show()
