from autograd import numpy as np
from matplotlib import pyplot as plt

from lab4.grad_descent import fit_model as simple_grad_descent
from lab4.impulse_descent import fit_model as impulse_grad_descent
from lab4.adam import fit_model as adam_descent


def f(w: np.ndarray):
    x, y = w
    return np.square(1 - x) + 100 * (np.square(y - np.square(x)))


EPOCHS = 10
ALPHA = 0.002
BETA = 0.05
P1 = 0.9
P2 = 0.999
EPSILON = 1 * 10 ** -8
FEATURES = (-1.5, 2.5)

x = np.linspace(-2, 3, 200)
y = np.linspace(-2, 3, 200)

X, Y = np.meshgrid(x, y)
Z = f((X, Y))

for descent, args in (
        (simple_grad_descent, (f, EPOCHS, ALPHA, FEATURES)),
        (impulse_grad_descent, (f, EPOCHS, ALPHA, BETA, FEATURES)),
        (adam_descent, (f, EPOCHS, 1, P1, P2, EPSILON, FEATURES))
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.2)

    steps = descent(*args)

    xs = steps[:, 0]
    ys = steps[:, 1]
    zs = f((xs, ys))

    ax.plot(xs, ys, zs, color="red", marker="o")
    for x_i, y_i, z_i in zip(xs, ys, zs):
        ax.text(x_i, y_i, z_i + 1, f"({x_i:.2f}, {y_i:.2f})")

    plt.show()
