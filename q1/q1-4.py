import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error


def f(x):
    return x + np.sin(1.5 * x)


def generate_data(num_dataset, x, n_samples):
    dataset = []
    while len(dataset) < num_dataset:
        y_values = f(x) + np.random.normal(0, np.sqrt(0.3), n_samples)
        dataset.append(np.column_stack((x, y_values)))

    return np.array(dataset)


if __name__ == '__main__':
    np.random.seed(42)

    x = np.array([np.random.uniform(0, 10) for _ in range(50)])
    datasets = generate_data(100, x, 50)

    X_test = datasets[0][:, 0][40:]
    y_true_test = f(X_test)
    mse_scores = np.zeros((100, 15))

    # Fit the estimators and evaluate them using the test set
    predictions = defaultdict(list)
    for degree in range(1, 16):
        poly_features = PolynomialFeatures(degree=degree)
        for i, dataset in enumerate(datasets):
            x_train = dataset[:, 0][:40].reshape(-1, 1)
            y_train = dataset[:, 1][:40]
            y_test = dataset[:, 1][40:]

            x_train_poly = poly_features.fit_transform(x_train)

            # fit a model on the current training set
            model = LinearRegression()
            model.fit(x_train_poly, y_train)

            # test the model on the test_set
            x_test_poly = poly_features.transform(X_test.reshape(-1, 1))
            y_pred = model.predict(x_test_poly)

            mse_scores[i, degree - 1] = mean_squared_error(y_test, y_pred)

            predictions[degree].append(y_pred)

    # calculate the squared bias, variance, and error
    squared_bias = []
    variance = []

    for degree in range(1, 16):
        mean_predictions = np.mean(np.array(predictions[degree]), axis=0)
        squared_bias.append(np.mean((y_true_test - mean_predictions) ** 2))
        variance.append(np.mean(np.var(np.array(predictions[degree]), axis=0)))

    # Plot the bias, variance, and error statistics
    degrees = np.arange(1, 16)
    plt.figure(figsize=(10, 6))

    plt.plot(degrees, squared_bias, label='Mean Squared Bias', marker='o')
    plt.plot(degrees, variance, label='Mean Variance', marker='o')
    plt.plot(degrees, np.mean(mse_scores, axis=0), label='Mean Squared Error (MSE)', marker='o')

    # Adding titles and labels
    plt.title('Squared Bias, Variance, and Error vs. Model Complexity (Degree)')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Error Metrics')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.savefig('q1-4.png', dpi=300, bbox_inches='tight')
    plt.show()



