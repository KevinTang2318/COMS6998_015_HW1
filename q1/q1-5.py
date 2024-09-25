import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
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
    predictions = []
    l2_predictions = []
    poly_features = PolynomialFeatures(degree=10)
    for train_set in train_datasets:
        x_train = train_set[:, 0].reshape(-1, 1)
        y_train = train_set[:, 1]
        x_train_poly = poly_features.fit_transform(x_train)

        # fit a model on the current training set
        model = LinearRegression()
        model.fit(x_train_poly, y_train)

        l2_model = Ridge(alpha=1.0)
        l2_model.fit(x_train_poly, y_train)

        # test the model on the test_set
        x_test_poly = poly_features.transform(x_test)
        y_pred = model.predict(x_test_poly)
        y_pred_l2 = l2_model.predict(x_test_poly)

        predictions.append(y_pred)
        l2_predictions.append(y_pred_l2)

    # calculate the squared bias, variance, and error
    mean_predictions = np.mean(predictions, axis=0)
    squared_bias = np.mean((y_test - mean_predictions) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    mse = mean_squared_error(y_test, mean_predictions)

    l2_mean_predictions = np.mean(l2_predictions, axis=0)
    l2_squared_bias = np.mean((y_test - l2_mean_predictions) ** 2)
    l2_variance = np.mean(np.var(l2_predictions, axis=0))
    l2_mse = mean_squared_error(y_test, l2_mean_predictions)

    print(f"Original model squared bias = {squared_bias}, Regularized model squared bias = {l2_squared_bias}")
    print(f"Original model variance = {variance}, Regularized model variance = {l2_variance}")
    print(f"Original model MSE = {mse}, Regularized model MSE = {l2_mse}")


