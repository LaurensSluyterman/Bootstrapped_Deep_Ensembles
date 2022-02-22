import numpy as np
import sys
sys.path.append('/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2020/Bootstrapped-Deep-Ensembles')
from neural_networks import LikelihoodNetwork, gen_deep_ensemble, gen_extra_deep_ensemble
from sklearn.model_selection import train_test_split
from utils import load_data
from target_simulation import create_true_function_and_variance

data_directory = 'Concrete'
X, Y = load_data(data_directory)
f, var_true = create_true_function_and_variance(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

N = 50
n_hidden = np.array([40, 30, 20])
predictions_random_training_random_targets = np.zeros((N, len(X_test)))
predictions_random_training_fixed_targets = np.zeros((N, len(X_test)))
difference_after_retraining = np.zeros((N, len(X_test)))

Y_train = np.random.normal(f(X_train), np.sqrt(var_true(X_train)))
same_targets_ensemble = gen_deep_ensemble(X_train, Y_train, N, hidden_units=n_hidden, n_epochs=40, save_epoch=28)
retrained_ensemble = gen_extra_deep_ensemble(X_train, same_targets_ensemble, 12)
for i in range(0, N):
    print(i)
    Y_train_new = np.random.normal(f(X_train), np.sqrt(var_true(X_train)))
    model = LikelihoodNetwork(X_train, Y_train_new, n_hidden=n_hidden, n_epochs=40)
    predictions_random_training_random_targets[i] = model.f(X_test)
    predictions_random_training_fixed_targets[i] = same_targets_ensemble[i].f(X_test)
    difference_after_retraining[i] = predictions_random_training_fixed_targets[i] - retrained_ensemble[i].f(X_test)


sigma2_d = np.mean(difference_after_retraining**2, axis=0)
sigma2_t = np.var(predictions_random_training_fixed_targets, axis=0)
sigma2_d_t = np.var(predictions_random_training_random_targets, axis=0)

print(np.mean(sigma2_d_t), np.mean(sigma2_t), np.mean(sigma2_d))
