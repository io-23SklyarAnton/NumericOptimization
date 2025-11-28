from autograd import numpy as np
from matplotlib import pyplot as plt

from lab2.grad_descent import fit_model


def f(x: float | np.ndarray):
    return (np.power(x, 3)) - (3 * np.square(x)) + 2


ALPHAS = [0.05, 0.1, 0.2]
START_XS = [0.0, 4.0, 3.0]
EPOCHS = [10, 5, 3]

for alpha, start_x, epochs in zip(ALPHAS, START_XS, EPOCHS):
    x_steps = fit_model(
        function=f,
        alpha=alpha,
        x=start_x,
        epochs=epochs
    )
    y_steps = f(x_steps)

    features = np.linspace(0, 4, 100)
    labels = f(features)
    plt.plot(features, labels)
    plt.plot(x_steps, y_steps, 'o--')
    plt.show()
