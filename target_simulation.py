from sklearn.ensemble import RandomForestRegressor
from neural_networks import LikelihoodNetwork
import numpy as np


def create_true_function_and_variance(X, Y):
    """Create a true function and variance using a Random Forest."""
    regr = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=2)
    regr.fit(X, Y)

    def f(x):
        return regr.predict(x)

    residuals_squared = (regr.predict(X) - Y) ** 2
    regr2 = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=2)
    regr2.fit(X, residuals_squared)

    def var_true(x):
        return regr2.predict(x)

    return f, var_true


def create_true_function_and_variance_NN(X, Y, n_hidden=np.array([40, 30, 20]), n_epochs=40):
    """Create a true function and variance using a neural network"""
    np.random.seed(2)
    network = LikelihoodNetwork(X, Y, n_hidden=n_hidden, n_epochs=n_epochs, verbose=False)

    def f(x):
        return network.f(x)

    def var_true(x):
        return network.sigma(x) ** 2

    return f, var_true


def gen_new_targets(X, f, var_true, dist="Gaussian"):
    """Generates new targets for given covariates, true function and variance."""
    if dist == "Gaussian":
        return np.random.normal(loc=f(X), scale=np.sqrt(var_true(X)))
    if dist == "t3":
        return f(X) + np.sqrt((1 / 3)) * np.sqrt(var_true(X)) * np.random.standard_t(df=3, size=len(X))
    if dist == 'gamma':
        k = 1 / 10
        theta = np.sqrt(10)
        return f(X) + np.sqrt(var_true(X)) * (np.random.gamma(shape=k, scale=theta, size=len(X)) - k * theta)
