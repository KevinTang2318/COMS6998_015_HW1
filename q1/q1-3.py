import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def f(x):
    return x + np.sin(1.5 * x)


def generate_data():
    x_values = np.random.uniform(0, 10, 20)
    y_values = f(x_values) + np.random.normal(0, np.sqrt(0.3), size=20)

    return x_values, y_values


if __name__ == '__main__':
    np.random.seed(42)
    x, y = generate_data()
    x = x.reshape(-1, 1)

    degrees = [1, 3, 10]
    models = {}

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x)

        model = LinearRegression()
        model.fit(x_poly, y)
        models[degree] = model

    # plot original model
    plt.scatter(x, y, color='blue', label='Sampled Data')
    x_plot = np.linspace(0, 10, 400).reshape(-1, 1)
    plt.plot(x_plot, f(x_plot), label='True Function f(x)', linestyle='--', color='red')

    # Plot each estimator
    for degree, model in models.items():
        poly_features = PolynomialFeatures(degree=degree)
        x_poly_plot = poly_features.fit_transform(x_plot)
        y_poly_plot = model.predict(x_poly_plot)
        plt.plot(x_plot, y_poly_plot, label=f'g_{degree}(x) with degree {degree}')

    plt.legend()
    plt.title('Polynomial Estimators')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('q1-3.png', dpi=300, bbox_inches='tight')
    plt.show()


