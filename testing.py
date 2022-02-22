
import keras.backend as K
from keras import initializers
from tensorflow.keras.layers import InputSpec
from keras.layers import Wrapper
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = (K.sigmoid(self.p_logit))

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - (K.sigmoid(self.p_logit)))
        dropout_regularizer = (K.sigmoid(self.p_logit)) * K.log((K.sigmoid(self.p_logit)))
        dropout_regularizer += (1. - (K.sigmoid(self.p_logit))) * K.log(1. - (K.sigmoid(self.p_logit)))
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log((K.sigmoid(self.p_logit)) + eps)
            - K.log(1. - (K.sigmoid(self.p_logit)) + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - (K.sigmoid(self.p_logit))
        x *= random_tensor
        x /= retain_prob
        return x

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)

def heteroscedastic_loss(true, pred):
    mean = pred[:, :1]
    log_var = pred[:, 1:]
    precision = K.exp(-log_var)
    return K.sum(precision * (true - mean) ** 2. + log_var, -1)
N = X_train.shape[0]
D = 1
l = 1e-4
wd = l ** 2. / N
dd = 2. / N
N = X_train.shape[0]
D = 1
l = 1e-4
wd = l ** 2. / N
dd = 2. / N
nb_features = 5
inp = Input(shape=(1,))
x = inp
x = ConcreteDropout(Dense(nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
x = ConcreteDropout(Dense(nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
x = ConcreteDropout(Dense(nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
mean = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x)
log_var = ConcreteDropout(Dense(D), weight_regularizer=wd, dropout_regularizer=dd)(x)
# out = Dense(2)(x)
out = tf.keras.layers.Concatenate()([mean, log_var])
model = Model(inp, out)


model.compile(optimizer='adam', loss=heteroscedastic_loss)
model.fit(X_train, Y_train, epochs=200, batch_size=32, verbose=1)

model.predict(X_train)

X_train = np.random.uniform(-1,1, 5000)
Y_train = np.random.normal(loc=2*X_train**2, scale=0.1)

import matplotlib.pyplot as plt

X = np.sort(X_train)
plt.plot(X, model.predict(X)[:,0], 'o')
plt.show()



data_directory = 'bostonHousing'
simulation_method = 'RF'
n_epochs = 40
distribution = 'Gaussian'
n_hidden = np.array([40, 30, 20])