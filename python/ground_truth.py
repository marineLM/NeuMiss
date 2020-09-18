import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import norm
from math import sqrt, floor, log
from joblib import Memory
from scipy.optimize import root_scalar
from scipy.special import comb
from amputation import MCAR, MAR_logistic

location = './cachedir'
memory = Memory(location, verbose=0)


def gen_params_selfmasking(n_features, missing_rate, prop_latent, sm_type,
                           sm_param, snr, perm=False, link='linear',
                           random_state=None):
    """Creates parameters for generating data with `generate_data_selfmasked`.

    Parameters
    ----------
    n_features: int
        The number of features desired.

    missing_rate: float
        The percentage of missing entries for each incomplete feature.
        Entries should be between 0 and 1.

    prop_latent: float
        The number of latent factors used to generate the covariance matrix is
        prop_latent*n_feature. The less factors the higher the correlations.
        Should be between 0 and 1.

    sm_type: str
        Type of selfmasking function used. One of `gaussian` or `probit`.

    sm_param: float
        Parameter for the selfmasking function.

        - If `sm_type == 'gaussian'`, then `sm_param` is the parameter called
        `k` in the paper that controls the mean of the Gaussian selfmasking
        function.

        - If `sm_type == 'probit'`, then `sm_param`is the parameter called
        `lambda`in the paper that controls the slope of the probit selfmasking
        function.

    snr: float
        The desired signal to noise ratio.

    perm: bool
        If perm is False, then selfmasking is performed. Otherwise, masking is
        performed based on another variable, chosen randomly, then the actual
        variable to be masked.

    link: str
        If 'linear', then the link function between y and X and linear. If
        'nonlinear', the link functionis chosen as y + y^3.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    if missing_rate > 1 or missing_rate < 0:
        raise ValueError("missing_rate must be >= 0 and <= 1, got %s" %
                         missing_rate)

    if prop_latent > 1 or prop_latent < 0:
        raise ValueError("prop_latent should be between 0 and 1")

    rng = check_random_state(random_state)

    # Generate covariance and mean
    # ---------------------------
    B = rng.randn(n_features, int(prop_latent*n_features))
    cov = B.dot(B.T) + np.diag(
        rng.uniform(low=0.01, high=0.1, size=n_features))

    mean = rng.randn(n_features)

    # Adapt the remaining parameters of the selfmasking function to obtain the
    # desired missing rate
    # ---------------------
    sm_params = {}

    if sm_type == 'probit':
        lam = sm_param
        sm_params['lambda'] = lam
        sm_params['c'] = np.zeros(n_features)
        for i in range(n_features):
            sm_params['c'][i] = lam*(mean[i] - norm.ppf(missing_rate)*np.sqrt(
                1/lam**2+cov[i, i]))

    elif sm_type == 'gaussian':
        k = sm_param
        sm_params['k'] = k
        sm_params['sigma2_tilde'] = np.zeros(n_features)

        min_x = missing_rate**2/(1-missing_rate**2)

        def f(x):
            y = -2*(1+x)*log(missing_rate*sqrt(1/x+1))
            return y

        for i in range(n_features):
            max_x = min_x
            while f(max_x) < k**2:
                max_x += 1
            sol = root_scalar(lambda x: f(x) - k**2, method='bisect',
                              bracket=(max_x-1, max_x), xtol=1e-3)

            sm_params['sigma2_tilde'][i] = sol.root*cov[i, i]

    # Generate beta
    beta = np.repeat(1., n_features + 1)

    # Convert the desired signal-to-noise ratio to a noise variance
    # var_Y = beta[1:].dot(cov).dot(beta[1:])
    # sigma2_noise = var_Y/snr

    return (n_features, sm_type, sm_params, mean, cov, beta, snr,
            perm, link)


def gen_data_selfmasking(n_sizes, data_params, random_state=None):

    rng = check_random_state(random_state)

    n_features, sm_type, sm_params, mean, cov, beta, snr, perm, link = data_params

    X = np.empty((0, n_features))
    y = np.empty((0, ))
    current_size = 0

    if perm:
        perms = rng.permutation(n_features)

    for _, n_samples in enumerate(n_sizes):

        current_X = rng.multivariate_normal(
                mean=mean, cov=cov,
                size=n_samples-current_size,
                check_valid='raise')

        current_y = beta[0] + current_X.dot(beta[1:])

        if link == 'nonlinear':
            current_y /= sqrt(np.var(current_y))
            current_y = current_y + current_y**3

        var_y = np.mean((current_y - np.mean(current_y))**2)
        sigma2_noise = var_y/snr

        noise = rng.normal(
            loc=0, scale=sqrt(sigma2_noise), size=n_samples-current_size)
        current_y += noise

        current_M = np.zeros((n_samples-current_size, n_features))
        for j in range(n_features):
            X_j = current_X[:, j]
            if sm_type == 'probit':
                lam = sm_params['lambda']
                c = sm_params['c'][j]
                prob = norm.cdf(lam*X_j - c)
            elif sm_type == 'gaussian':
                k = sm_params['k']
                sigma2_tilde = sm_params['sigma2_tilde'][j]
                mu_tilde = mean[j] + k*sqrt(cov[j, j])
                prob = np.exp(-0.5*(X_j - mu_tilde)**2/sigma2_tilde)

            current_M[:, j] = rng.binomial(n=1, p=prob, size=len(X_j))

        if not perm:
            np.putmask(current_X, current_M, np.nan)
        else:
            for j in range(n_features):
                new_j = perms[j]
                np.putmask(current_X[:, new_j], current_M[:, j], np.nan)

        X = np.vstack((X, current_X))
        y = np.hstack((y, current_y))

        current_size = n_samples

        yield X, y


def gen_params(n_features, missing_rate, prop_latent, snr, masking,
               prop_for_masking=None, random_state=None):
    """Creates parameters for generating multivariate Gaussian data.

    Parameters
    ----------
    n_features: int
        The number of features desired.

    missing_rate: float
        The percentage of missing entries for each incomplete feature.
        Entries should be between 0 and 1.

    prop_latent: float
        The number of latent factors used to generate the covariance matrix is
        prop_latent*n_feature. The less factors the higher the correlations.
        Should be between 0 and 1.

    snr: float
        The desired signal to noise ratio.

    masking: str
        The desired masking type. One of 'MCAR', 'MAR_logistic'.

    prop_for_masking: float, default None
        The proportion of variables used in the logistic function for masking.
        It is not relevant if `masking == 'MCAR'` or
        `masking == 'MNAR_logistic'`.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    if missing_rate > 1 or missing_rate < 0:
        raise ValueError("missing_rate must be >= 0 and <= 1, got %s" %
                         missing_rate)

    if prop_latent > 1 or prop_latent < 0:
        raise ValueError("prop_latent should be between 0 and 1")

    if prop_for_masking and (prop_for_masking > 1 or prop_for_masking < 0):
        raise ValueError("prop_for_masking should be between 0 and 1")

    rng = check_random_state(random_state)

    # Generate covariance and mean
    # ---------------------------
    B = rng.randn(n_features, int(prop_latent*n_features))
    cov = B.dot(B.T) + np.diag(
        rng.uniform(low=0.01, high=0.1, size=n_features))

    mean = rng.randn(n_features)

    # Generate beta
    beta = np.repeat(1., n_features + 1)

    # Convert the desired signal-to-noise ratio to a noise variance
    var_Y = beta[1:].dot(cov).dot(beta[1:])
    sigma2_noise = var_Y/snr

    return (n_features, mean, cov, beta, sigma2_noise, masking, missing_rate,
            prop_for_masking)


def gen_data(n_sizes, data_params, random_state=None):

    rng = check_random_state(random_state)

    (n_features, mean, cov, beta, sigma2_noise, masking, missing_rate,
     prop_for_masking) = data_params

    X = np.empty((0, n_features))
    y = np.empty((0, ))
    current_size = 0

    for _, n_samples in enumerate(n_sizes):

        current_X = rng.multivariate_normal(
                mean=mean, cov=cov,
                size=n_samples-current_size,
                check_valid='raise')

        noise = rng.normal(
            loc=0, scale=sqrt(sigma2_noise), size=n_samples-current_size)
        current_y = beta[0] + current_X.dot(beta[1:]) + noise

        if masking == 'MCAR':
            current_M = MCAR(current_X, missing_rate, rng)
        elif masking == 'MAR_logistic':
            current_M = MAR_logistic(current_X, missing_rate, prop_for_masking,
                                     rng)

        np.putmask(current_X, current_M, np.nan)

        X = np.vstack((X, current_X))
        y = np.hstack((y, current_y))

        current_size = n_samples

        yield X, y


class BayesPredictor_MCAR_MAR():
    """This is the Bayes predicor for multivariate Gaussian data and MCAR or
    MAR missing data mechanisms."""

    def __init__(self, data_params):
        self.data_params = data_params

    def fit(self, X, y):
        return self

    def predict(self, X):
        _, mu, sigma, beta, _, _, _, _ = self.data_params

        pred = []
        for x in X:
            m = ''.join([str(mj) for mj in np.isnan(x).astype(int)])

            obs = np.where(np.array(list(m)).astype(int) == 0)[0]
            mis = np.where(np.array(list(m)).astype(int) == 1)[0]

            predx = beta[0]
            if len(mis) > 0:
                predx += beta[mis + 1].dot(mu[mis])
            if len(obs) > 0:
                predx += beta[obs + 1].dot(x[obs])
            if len(obs) * len(mis) > 0:
                sigma_obs = sigma[np.ix_(obs, obs)]
                sigma_obs_inv = np.linalg.inv(sigma_obs)
                sigma_misobs = sigma[np.ix_(mis, obs)]

                predx += beta[mis + 1].dot(sigma_misobs).dot(
                    sigma_obs_inv).dot(x[obs] - mu[obs])

            pred.append(predx)

        return np.array(pred)


class BayesPredictor_gaussian_selfmasking():
    """This is the Bayes predicor for Gaussian data with a Gaussian
    selfmasking missing data mechanism"""

    def __init__(self, data_params):

        _, sm_type, _, _, _, _, _, perm, _ = data_params

        if sm_type == 'probit':
            raise ValueError('This Bayes predictor is only valid for' +
                             'Gaussian selfmasking and not probit selfmasking')

        if perm:
            raise ValueError('The Bayes predictor is not available for' +
                             'perm = True')

        self.data_params = data_params

    def fit(self, X, y):
        return self

    def predict(self, X):

        _, _, sm_params, mu, cov, beta, _, _, _ = self.data_params

        k = sm_params['k']
        tsigma2 = sm_params['sigma2_tilde']
        tmu = mu + k*np.sqrt(np.diag(cov))

        pred = []
        for x in X:
            mis = np.where(np.isnan(x))[0]
            obs = np.where(~np.isnan(x))[0]

            D_mis_inv = np.diag(1/tsigma2[mis])

            cov_misobs = cov[np.ix_(mis, obs)]
            cov_obs_inv = np.linalg.inv(cov[np.ix_(obs, obs)])
            cov_mis = cov[np.ix_(mis, mis)]

            mu_mis_obs = mu[mis] + cov_misobs.dot(cov_obs_inv).dot(
                x[obs] - mu[obs])
            cov_mis_obs = cov_mis - cov_misobs.dot(cov_obs_inv).dot(
                cov_misobs.T)
            cov_mis_obs_inv = np.linalg.inv(cov_mis_obs)

            predx = beta[0]
            predx += beta[obs + 1].dot(x[obs])
            S = np.linalg.inv(D_mis_inv + cov_mis_obs_inv)
            predx += beta[mis + 1].dot(
                S.dot(D_mis_inv.dot(tmu[mis]) +
                      cov_mis_obs_inv.dot(mu_mis_obs))
            )

            pred.append(predx)

        return np.array(pred)
