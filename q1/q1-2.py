import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def f(x):
    return x + np.sin(1.5 * x)


x_values = np.random.uniform(0, 10, 20)

y_values = f(x_values) + np.random.normal(0, np.sqrt(0.3), size=20)

x_smooth = np.linspace(0, 10, 400)
f_smooth = f(x_smooth)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', label='Generated Data (y)')
plt.plot(x_smooth, f_smooth, color='red', label='f(x) = x + sin(1.5x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated dataset and smooth line plot of f(x)')
plt.legend()
plt.grid(True)
plt.savefig('q1-2.png', dpi=300, bbox_inches='tight')
plt.show()
