from autograd import numpy as np
from matplotlib import pyplot as plt

from lab4.grad_descent import fit_model as simple_grad_descent
from lab4.impulse_descent import fit_model as impulse_grad_descent
from lab4.adam import fit_model as adam_descent


def f(w: np.ndarray):
    x, y = w
    return np.square(1 - x) + 100 * np.square(y - np.square(x))


EPOCHS = 40
ALPHA = 0.0025
BETA = 0.05
P1 = 0.9
P2 = 0.999
EPSILON = 1e-8
FEATURES = (-1.5, 2.5)

x = np.linspace(-2, 3, 200)
y = np.linspace(-2, 3, 200)
X, Y = np.meshgrid(x, y)
Z = f((X, Y))

w_minimum = np.array([1.0, 1.0])

optimizers = [
    ("Simple GD", simple_grad_descent, (f, EPOCHS, ALPHA, FEATURES)),
    ("Momentum", impulse_grad_descent, (f, EPOCHS, ALPHA, BETA, FEATURES)),
    ("Adam", adam_descent, (f, EPOCHS, 0.9, P1, P2, EPSILON, FEATURES)),
]

for title, descent, args in optimizers:
    steps = descent(*args)
    xs = steps[:, 0]
    ys = steps[:, 1]
    zs = f((xs, ys))
    values = zs
    dists = np.linalg.norm(steps - w_minimum, axis=1)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("f(x, y)")
    ax3d.plot_surface(X, Y, Z, cmap="plasma", alpha=0.2, edgecolor="none")
    ax3d.plot(xs, ys, zs, color="red", marker="o")

    z_min = f(w_minimum)
    ax3d.scatter([w_minimum[0]], [w_minimum[1]], [z_min], color="green", marker="*", s=120)

    axc = fig.add_subplot(2, 2, 2)
    axc.set_xlabel("X")
    axc.set_ylabel("Y")
    cs = axc.contour(X, Y, Z, levels=70, cmap="plasma")
    axc.clabel(cs, inline=True, fontsize=7)
    axc.plot(xs, ys, color="red", marker="o", label="trajectory")
    axc.scatter([w_minimum[0]], [w_minimum[1]], color="green", marker="*", s=120, label="min (1,1)")
    axc.legend(fontsize=8)

    axf = fig.add_subplot(2, 1, 2)
    axf.set_xlabel("iteration")
    axf.set_ylabel("f(w)")
    axf.plot(range(len(values)), values, marker="o")
    axf.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()
