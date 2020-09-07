'''This file implements amputation procedures according to various missing
data mechanisms. It was inspired from
https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
'''

import numpy as np
from sklearn.utils import check_random_state
from scipy.optimize import fsolve


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MCAR(X, p, random_state):
    """
    Missing completely at random mechanism.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    ber = rng.rand(n, d)
    mask = ber < p

    return mask


def MAR_logistic(X, p, p_obs, random_state):
    """
    Missing at random mechanism with a logistic masking model. First, a subset
    of variables with *no* missing values is randomly selected. The remaining
    variables have missing values according to a logistic model with random
    weights, but whose intercept is chosen so as to attain the desired
    proportion of missing values on those variables.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for
        the logistic masking model.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    # number of variables that will have no missing values
    # (at least one variable)
    d_obs = max(int(p_obs * d), 1)
    # number of variables that will have missing values
    d_na = d - d_obs

    # Sample variables that will all be observed, and those with missing values
    idxs_obs = rng.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Other variables will have NA proportions that depend on those observed
    # variables, through a logistic model. The parameters of this logistic
    # model are random, and adapted to the scale of each variable.
    mu = X.mean(0)
    cov = (X-mu).T.dot(X-mu)/n
    cov_obs = cov[np.ix_(idxs_obs, idxs_obs)]
    coeffs = rng.randn(d_obs, d_na)
    v = np.array([coeffs[:, j].dot(cov_obs).dot(
        coeffs[:, j]) for j in range(d_na)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d_na)
    coeffs /= steepness*np.sqrt(v)

    # Move the intercept to have the desired amount of missing values
    intercepts = np.zeros((d_na))
    for j in range(d_na):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(X[:, idxs_obs].dot(w) + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(X[:, idxs_obs].dot(coeffs) + intercepts)
    ber = rng.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask
