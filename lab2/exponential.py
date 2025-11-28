from autograd import numpy as np
from matplotlib import pyplot as plt

from lab2.grad_descent import fit_model


def f(x: float | np.ndarray):
    return np.square(x + 1) + 1


ALPHAS = [0.1, 0.3, 1]
START_XS = [-3.0, 1.0, 0.0]
EPOCHS = [7, 4, 5]

for alpha, start_x, epochs in zip(ALPHAS, START_XS, EPOCHS):
    x_steps = fit_model(
        function=f,
        alpha=alpha,
        x=start_x,
        epochs=epochs
    )
    y_steps = f(x_steps)

    features = np.linspace(-3, 1, 100)
    labels = f(features)
    plt.plot(features, labels)
    plt.plot(x_steps, y_steps, 'ro')
    plt.show()
