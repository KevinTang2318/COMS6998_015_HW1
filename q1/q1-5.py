import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


def f(x):
    return x + np.sin(1.5 * x)


def generate_data(num_dataset, x):
    dataset = []
    while len(dataset) < num_dataset:
        y_values = f(x) + np.random.normal(0, np.sqrt(0.3))
        dataset.append(np.column_stack((x, y_values)))

    return np.array(dataset)


if __name__ == '__main__':
    np.random.seed(42)

    x = np.array([np.random.uniform(0, 10) for _ in range(50)])
    datasets = generate_data(100, x)
    X_test = datasets[0][:, 0][40:]
    y_true_test = f(X_test)

    # Fit the estimators and evaluate them using the test set
    predictions = []
    mses = []

    l2_predictions = []
    l2_mses = []
    poly_features = PolynomialFeatures(degree=10)
    for dataset in datasets:
        x_train = dataset[:, 0][:40].reshape(-1, 1)
        y_train = dataset[:, 1][:40]
        y_test = dataset[:, 1][40:]
        x_train_poly = poly_features.fit_transform(x_train)

        # fit a model on the current training set
        model = LinearRegression()
        model.fit(x_train_poly, y_train)

        l2_model = Ridge(alpha=1.0)
        l2_model.fit(x_train_poly, y_train)

        # test the model on the test_set
        x_test_poly = poly_features.transform(X_test.reshape(-1, 1))
        y_pred = model.predict(x_test_poly)
        y_pred_l2 = l2_model.predict(x_test_poly)

        predictions.append(y_pred)
        l2_predictions.append(y_pred_l2)

        mses.append(mean_squared_error(y_test, y_pred))
        l2_mses.append(mean_squared_error(y_test, y_pred_l2))

    # calculate the squared bias, variance, and error
    mean_predictions = np.mean(predictions, axis=0)
    squared_bias = np.mean((y_true_test - mean_predictions) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    mse = np.mean(mses)

    l2_mean_predictions = np.mean(l2_predictions, axis=0)
    l2_squared_bias = np.mean((y_true_test - l2_mean_predictions) ** 2)
    l2_variance = np.mean(np.var(l2_predictions, axis=0))
    l2_mse = np.mean(l2_mses)

    print(f"Original model squared bias = {squared_bias}, Regularized model squared bias = {l2_squared_bias}")
    print(f"Original model variance = {variance}, Regularized model variance = {l2_variance}")
    print(f"Original model MSE = {mse}, Regularized model MSE = {l2_mse}")


