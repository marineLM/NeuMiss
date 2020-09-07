"""This file contains the estimators:
    - ConstantImputedLR
    - ExpandedLR
    - EMLR
    - ConstantImputedMLPR

"""
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from rpy2.robjects import Matrix, numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
norm = importr("norm")

numpy2ri.activate()
pandas2ri.activate()


class ConstantImputedLR():
    def __init__(self):
        self._reg = LinearRegression()

    def transform(self, X):
        T = X.copy()
        M = np.isnan(T)
        np.putmask(T, M, 0)
        T = np.hstack((T, M))
        return T

    def fit(self, X, y):
        T = self.transform(X)
        self._reg.fit(T, y)
        return self

    def predict(self, X):
        T = self.transform(X)
        return self._reg.predict(T)


class EMLR(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        Z = np.hstack((y[:, np.newaxis], X))
        s = norm.prelim_norm(Z)
        thetahat = norm.em_norm(s, showits=False,
                                criterion=np.sqrt(np.finfo(float).eps))
        parameters = norm.getparam_norm(s, thetahat)
        self.mu_joint = np.array(parameters[0])
        self.Sigma_joint = np.array(parameters[1])

    def predict(self, X):
        # raw
        pred = np.empty(X.shape[0])
        for i, x in enumerate(X):
            indices = np.where(~np.isnan(x))[0] + 1
            x_obs = x[~np.isnan(x)]

            mu_X = self.mu_joint[indices]
            Sigma_X = self.Sigma_joint[np.ix_(indices, indices)]
            mu_y = self.mu_joint[0]
            Sigma_yX = self.Sigma_joint[0, indices]

            if len(indices) == 0:
                pred[i] = mu_y
            elif len(indices) == 1:
                beta = (
                    mu_y - Sigma_yX * mu_X / Sigma_X,
                    Sigma_yX / Sigma_X)
                pred[i] = beta[0] + beta[1] * x_obs
            else:
                beta = (
                    mu_y - Sigma_yX.dot(np.linalg.inv(Sigma_X)).dot(mu_X),
                    Sigma_yX.dot(np.linalg.inv(Sigma_X)))
                pred[i] = beta[0] + beta[1].dot(x_obs)
        return pred


class MICELR():
    def __init__(self):
        self._reg = LinearRegression()
        self._imp = IterativeImputer(random_state=0)

    def fit(self, X, y):
        T = self._imp.fit_transform(X)
        self._reg.fit(T, y)
        return self

    def predict(self, X):
        T = self._imp.transform(X)
        return self._reg.predict(T)
