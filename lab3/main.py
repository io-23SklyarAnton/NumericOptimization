from autograd import numpy as np
from matplotlib import pyplot as plt

from lab2 import grad_descent as simple_grad_descent
from lab3 import grad_descent as impulse_grad_descent


def f(x: float | np.ndarray):
    return (np.power(x - 1, 6)) + (0.5 * np.square(x - 1))


ALPHAS = [0.02, 0.01, 0.2]
START_XS = [-1.0, 3.0, 0.0]
EPOCHS = [7, 7, 7]
BETAS = [0.2, 0.3, 0.3]

for alpha, start_x, epochs, beta in zip(ALPHAS, START_XS, EPOCHS, BETAS):
    features = np.linspace(-1, 3, 100)
    labels = f(features)
    plt.plot(features, labels)

    x_steps = simple_grad_descent.fit_model(
        function=f,
        alpha=alpha,
        x=start_x,
        epochs=epochs
    )
    y_steps = f(x_steps)
    plt.plot(x_steps, y_steps, 'o--')

    x_steps = impulse_grad_descent.fit_model(
        function=f,
        alpha=alpha,
        x=start_x,
        epochs=epochs,
        beta=beta
    )
    y_steps = f(x_steps)
    plt.plot(x_steps, y_steps, 'g--')
    plt.show()
