import numpy as np

def normalize(X):
    mu = np.mean(X)
    # Find the standard deviation for each col
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma

    return X_norm

def normalize_single(x):
    sum = np.sum(x)
    mu = sum / len(x)

    s_sum = 0
    for x_i in x:
        s_sum += (x_i - mu) ** 2

    sigma = np.sqrt(s_sum / len(x))

    res = []
    for x_i in x:
        r = (x_i - mu) / sigma
        res.append(r)

    return res
