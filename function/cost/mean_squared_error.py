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


def run_linear():
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])

    cost = compute_cost(x, y, 200, 100)
    assert cost == 0.0


if __name__ == '__main__':
    run_linear()
