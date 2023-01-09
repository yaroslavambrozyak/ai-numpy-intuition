import numpy as np


def compute_cost(x, y, w, b):
    """
    Computes cost function for 1D array values
    :param x: ndarray (1,)
    :param y: ndarray(1,)
    :param w: scalar
    :param b: scalar
    :return: total cost
    """
    # get size of x array
    m = x.shape[0]
    sum_cost = 0
    for i in range(m):
        # get model prediction
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        sum_cost += cost

    total_cost = (1 / (2 * m)) * sum_cost
    return total_cost


def compute_cost_multiple(X, y, w, b):
    """
    Compute cost for multiple features
    :param X: ndarray(m,n)
    :param y: ndarray(m,)
    :param w: ndarray(m,)
    :param b: scalar
    :return: total cost
    """

    m = X.shape[0]
    sum_cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = (f_wb_i - y[i]) ** 2
        sum_cost += cost

    total_cost = sum_cost / (2 * m)

    return total_cost


def run_linear():
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])

    cost = compute_cost(x, y, 200, 100)
    assert cost == 0.0


def run_linear_multiple():
    X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y = np.array([460, 232, 178])

    w = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
    b = 785.1811367994083

    cost = compute_cost_multiple(X, y, w, b)
    assert cost == 1.5578904428966628e-12


if __name__ == '__main__':
    run_linear()
    run_linear_multiple()
