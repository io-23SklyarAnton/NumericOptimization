import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt


def math_function(x, y):
    return (np.square(x) + np.square(y)) * np.sin(x * y)


df_dx = egrad(math_function, 0)
df_dy = egrad(math_function, 1)

x, y = float(1), float(2)
print("df/dx:", df_dx(x, y))
print("df/dy:", df_dy(x, y))

x = np.linspace(-7, 7, 2000)
y = np.linspace(-7, 7, 2000)

X, Y = np.meshgrid(x, y)
Z = math_function(X, Y)
Zx = df_dx(X, Y)
Zy = df_dy(X, Y)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax.set_title(r"$f(x,y)=(x^2+y^2)\sin(xy)$")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

plt.show()

for Zv, title in [(Zx, r"$\partial f/\partial x$"), (Zy, r"$\partial f/\partial y$")]:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Zv, cmap="plasma", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel(title)
    plt.tight_layout()
    plt.show()