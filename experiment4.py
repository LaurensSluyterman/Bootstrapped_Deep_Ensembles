import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2020/Bootstrapped-Deep-Ensembles')
from neural_networks import gen_deep_ensemble, gen_extra_deep_ensemble
from intervals import PICI_deep_ensemble, PICI_bootstrapped_deep_ensemble, deep_ensemble_prediction
matplotlib.rcParams['text.usetex'] = True

np.random.seed(1)
x = np.linspace(0.1, 0.9, 7)
x_test = np.linspace(0, 1, 200)
y = np.random.normal(loc=0, scale=0.2, size=len(x))
n_hidden = np.array([400, 200, 100])
M = 5
n_epochs = 80
save_epoch = np.int(n_epochs * 0.7)
deep_networks = gen_deep_ensemble(x, y, M, n_hidden, n_epochs=n_epochs,
                                  save_epoch=save_epoch, verbose=True, reg=0)
extra_deep_networks = gen_extra_deep_ensemble(x, deep_networks,
                                              n_epochs=n_epochs-save_epoch)

PI_BDE, CI_BDE = PICI_bootstrapped_deep_ensemble(deep_networks, extra_deep_networks, x_test, 0.1)
PI_DE, CI_DE = PICI_deep_ensemble(deep_networks, x_test, 0.1)

fontsize = 17
linewidth=1.5
plt.plot(x_test, CI_BDE[:, 0], linestyle='dashed', color='r', linewidth=linewidth)
plt.plot(x_test, CI_BDE[:, 1], linestyle='dashed', color='r', linewidth=linewidth, label='BDE')
plt.plot(x_test, CI_DE[:, 0], linestyle='dotted', color='b', linewidth=linewidth)
plt.plot(x_test, CI_DE[:, 1], linestyle='dotted', color='b', label='DE', linewidth=linewidth)
plt.plot(x_test, deep_ensemble_prediction(deep_networks, x_test), label="$\hat{f}(x)$")
plt.plot(x, y, 'o')
plt.legend(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.locator_params(axis="y", nbins=6)
plt.locator_params(axis="x", nbins=5)
plt.xlabel('$x$', fontsize=fontsize)
plt.show()
