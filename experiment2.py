import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2020/Bootstrapped-Deep-Ensembles')
from neural_networks import LikelihoodNetwork
from sklearn.model_selection import train_test_split
from utils import load_data
from target_simulation import create_true_function_and_variance
matplotlib.rcParams['text.usetex'] = True

data_directory = 'protein-tertiary-structure'
X, Y = load_data(data_directory)
f, var_true = create_true_function_and_variance(X, Y)


def calculate_ensemble_variance(X, ensemble):
    """Calculate the variance of the predictions of  an ensemble of networks."""
    predictions = np.zeros((len(ensemble), len(X)))
    for i in range(len(ensemble)):
        predictions[i] = ensemble[i].f(X)
    return np.var(predictions, axis=0)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=5000,
                                                    random_state=1)
M = 100
training_sizes = [500, 5000, 10000, 20000, 30000]
n_hidden = np.array([40, 30, 20])
variance_random_targets = np.zeros(len(training_sizes))
variance_fixed_targets = np.zeros(len(training_sizes))
targets = np.random.normal(loc=f(X_train), scale=np.sqrt(var_true(X_train)))
n_epochs = 40
for i, N in enumerate(training_sizes):
    print(f'{i + 1} of {len(training_sizes)}')
    fixed_targets_ensemble = [LikelihoodNetwork(X_train[0:N], targets[0:N],
                                                n_hidden, n_epochs=n_epochs,
                                                verbose=False)
                              for _ in range(M)]
    random_targets_ensemble = [LikelihoodNetwork(X_train[0:N],
                                                 np.random.normal(f(X_train[0:N]),
                                                                  np.sqrt(var_true(X_train[0:N]))),
                                                 n_hidden, n_epochs=n_epochs,
                                                 verbose=False)
                               for _ in range(M)]
    variance_random_targets[i] = np.mean(calculate_ensemble_variance(X_test, random_targets_ensemble))
    variance_fixed_targets[i] = np.mean(calculate_ensemble_variance(X_test, fixed_targets_ensemble))

fontsize = 17
sigma2d = variance_random_targets - variance_fixed_targets
sigma2t = variance_fixed_targets

plt.plot(training_sizes, variance_fixed_targets, '--ro', label='$\sigma^{2}_{t}$')
plt.plot(training_sizes, sigma2d, '--bo', label='$\sigma^{2}_{d}$')
plt.legend(fontsize=fontsize)
plt.xlabel('N', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
