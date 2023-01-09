import numpy as np
from function.cost.mean_squared_error import compute_cost as mean_cost_f


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def compute_descent(x, y, w, b, alpha, num_iters, cost_function, gradient_function):
    j_history = []
    p_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        j_history.append(cost_function(x, y, w, b))
        p_history.append((w, b))

    return w, b, j_history, p_history


def run_linear():
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])

    w = 0
    b = 0
    iterations = 10_000
    alpha = 1.0e-2
    gradient_f = compute_gradient

    w, b, j_history, p_history = compute_descent(x, y, w, b, alpha, iterations, mean_cost_f, gradient_f)

    assert round(w, 4) == 199.9929
    assert round(b, 4) == 100.0116


if __name__ == '__main__':
    run_linear()
