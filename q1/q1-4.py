import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error

def f(x):
    return x + np.sin(1.5 * x)


def generate_data(num_data, test_set):

    dataset = []

    while len(dataset) < num_data:
        x_value = np.random.uniform(0, 10)
        y_value = f(x_value) + np.random.normal(0, np.sqrt(0.3))
        if [x_value, y_value] not in test_set:
            dataset.append([x_value, y_value])

    return np.array(dataset)


if __name__ == '__main__':
    np.random.seed(42)

    # Generate the test set shared across all 100 training sets
    test_data = generate_data(10, [])
    x_test = test_data[:, 0].reshape(-1, 1)
    y_test = test_data[:, 1]

    # Generate 100 train datasets that does not contain data from test set
    train_datasets = []
    for i in range(100):
        train_datasets.append(generate_data(40, test_data))

    # Fit the estimators and evaluate them using the test set
    predictions = defaultdict(list)
    for degree in range(1, 16):
        poly_features = PolynomialFeatures(degree=degree)
        for train_set in train_datasets:
            x_train = train_set[:, 0].reshape(-1, 1)
            y_train = train_set[:, 1]
            x_train_poly = poly_features.fit_transform(x_train)

            # fit a model on the current training set
            model = LinearRegression()
            model.fit(x_train_poly, y_train)

            # test the model on the test_set
            x_test_poly = poly_features.transform(x_test)
            y_pred = model.predict(x_test_poly)

            predictions[degree].append(y_pred)

    # calculate the squared bias, variance, and error
    squared_bias = []
    variance = []
    error = []

    for degree in range(1, 16):
        mean_predictions = np.mean(predictions[degree], axis=0)
        squared_bias.append(np.mean((y_test - mean_predictions) ** 2))
        variance.append(np.mean(np.var(predictions[degree], axis=0)))
        error.append(mean_squared_error(y_test, mean_predictions))

    # Plot the bias, variance, and error statistics
    degrees = np.arange(1, 16)
    plt.figure(figsize=(10, 6))

    plt.plot(degrees, squared_bias, label='Mean Squared Bias', marker='o')
    plt.plot(degrees, variance, label='Mean Variance', marker='o')
    plt.plot(degrees, error, label='Mean Squared Error (MSE)', marker='o')

    # Adding titles and labels
    plt.title('Squared Bias, Variance, and Error vs. Model Complexity (Degree)')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Error Metrics')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.savefig('q1-4.png', dpi=300, bbox_inches='tight')
    plt.show()



