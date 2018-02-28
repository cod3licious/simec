import numpy as np
np.random.seed(28)
import tensorflow as tf
tf.set_random_seed(28)
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.regularizers import Regularizer
from keras.losses import mean_squared_error


def center_K(K):
    # center the kernel matrix
    n, m = K.shape
    H = np.eye(n) - np.tile(1. / n, (n, n))
    B = np.dot(np.dot(H, K), H)
    return (B + B.T) / 2


def masked_mse(mask_value):
    """
    https://github.com/fchollet/keras/issues/7065
    compute mean squared error using only those values not equal to mask_value,
    e.g. to deal with missing values in the target similarity matrix
    """
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.sum(mask_true, axis=-1)
        return masked_mse
    f.__name__ = 'Masked MSE (mask_value={})'.format(mask_value)
    return f


class S_ll_reg(Regularizer):

    def __init__(self, s_ll_reg=0., S_ll=None, mask_value=None, l2_reg=0.):
        self.s_ll_reg = K.cast_to_floatx(s_ll_reg)
        if s_ll_reg > 0.:
            assert (S_ll is not None), "need to give S_ll"
            self.S_ll = S_ll
            if mask_value is None:
                self.errfun = mean_squared_error
            else:
                self.errfun = masked_mse(mask_value)
        else:
            self.S_ll = None
        self.l2_reg = K.cast_to_floatx(l2_reg)

    def __call__(self, x):
        regularization = 0.
        if self.s_ll_reg > 0.:
            regularization += self.s_ll_reg * self.errfun(self.S_ll, K.dot(K.transpose(x), x))
        if self.l2_reg > 0.:
            regularization += K.sum(self.l2_reg * K.square(x))
        return regularization

    def get_config(self):
        return {'s_ll_reg': float(self.s_ll_reg)}


class SimilarityEncoder(object):

    def __init__(self, in_dim, embedding_dim, out_dim, hidden_layers=[], sparse_inputs=False, mask_value=None,
                 opt=keras.optimizers.Adamax(lr=0.0005), l2_reg=0.00000001, l2_reg_emb=0.00001, l2_reg_out=0., s_ll_reg=0., S_ll=None):
        # checks for s_ll_regularization
        if s_ll_reg > 0.:
            assert S_ll is not None, "need S_ll"
            assert S_ll.shape == (out_dim, out_dim), "S_ll needs to be of shape (out_dim x out_dim)"
        # save some parameters we might need for later checks
        self.in_dim = in_dim
        self.out_dim = out_dim
        # inputs - might be sparse
        inputs = Input(shape=(in_dim,), sparse=sparse_inputs)
        # linear simec only gets the linear layer that maps to the embedding
        if not hidden_layers:
            embedding = Dense(embedding_dim, activation='linear',
                              kernel_regularizer=keras.regularizers.l2(l2_reg_emb))(inputs)
        else:
            # add additional hidden layers (first one acts on the input)
            # hidden_layers should be a list with (h_layer_dim, activation)
            for i, h in enumerate(hidden_layers):
                if i == 0:
                    x = Dense(h[0], activation=h[1],
                              kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
                else:
                    x = Dense(h[0], activation=h[1],
                              kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
            # after the hidden layers, add the embedding layer
            embedding = Dense(embedding_dim, activation='linear',
                              kernel_regularizer=keras.regularizers.l2(l2_reg_emb))(x)
        # add another linear layer to get the linear approximation of the target similarities
        outputs = Dense(out_dim, activation='linear',
                        kernel_regularizer=S_ll_reg(s_ll_reg, S_ll, mask_value, l2_reg_out))(embedding)
        # put it all into a model
        self.model = Model(inputs=inputs, outputs=outputs)
        # compile the model to minimize the MSE
        if mask_value is None:
            self.model.compile(optimizer=opt, loss='mse')
        else:
            self.model.compile(optimizer=opt, loss=masked_mse(mask_value))
        # placeholder for embedding model
        self.model_embed = None

    def fit(self, X, S, epochs=25, verbose=1):
        assert X.shape[1] == self.in_dim, "input dim of data doesn't match (%i != %i)" % (X.shape[1], self.in_dim)
        assert X.shape[0] == S.shape[0], "number of samples for inputs and targets doesn't match (%i != %i)" % (X.shape[0], S.shape[0])
        assert S.shape[1] == self.out_dim, "output dim of targets doesn't match (%i != %i)" % (S.shape[1], self.out_dim)
        self.model.fit(X, S, epochs=epochs, verbose=verbose)
        # store the model we need for the prediction
        self.model_embed = Sequential(self.model.layers[:-1])

    def transform(self, X):
        assert self.model_embed is not None, "need to fit model first"
        return self.model_embed.predict(X)
