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
    """
    Center the given square (and symmetric) kernel matrix

    Input:
        - K: square (and symmetric) kernel (similarity) matrix
    Returns:
        - centered kernel matrix (like if you had subtracted the mean from the input data)
    """
    n, m = K.shape
    assert n == m, "Kernel matrix needs to be square"
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


class LastLayerReg(Regularizer):

    def __init__(self, l2_reg=0., s_ll_reg=0., S_ll=None, orth_reg=0., embedding_dim=0, mask_value=None):
        """
        Custom regularizer used for the last layer of a SimEc
        """
        self.l2_reg = K.cast_to_floatx(l2_reg)
        self.s_ll_reg = K.cast_to_floatx(s_ll_reg)
        if s_ll_reg > 0.:
            assert (S_ll is not None), "need to give S_ll"
            self.S_ll = S_ll
        else:
            self.S_ll = None
        self.orth_reg = K.cast_to_floatx(orth_reg)
        if orth_reg > 0.:
            assert (embedding_dim > 0), "need to give shape of embedding layer, i.e. x.shape[0]"
            self.embedding_dim = embedding_dim
        if mask_value is None:
            self.errfun = mean_squared_error
        else:
            self.errfun = masked_mse(mask_value)

    def __call__(self, x):
        regularization = 0.
        if self.l2_reg > 0.:
            regularization += K.sum(self.l2_reg * K.square(x))
        if self.s_ll_reg > 0.:
            regularization += self.s_ll_reg * self.errfun(self.S_ll, K.dot(K.transpose(x), x))
        if self.orth_reg > 0.:
            regularization += self.orth_reg * self.errfun(K.eye(self.embedding_dim), K.dot(x, K.transpose(x)))
        return regularization

    def get_config(self):
        return {'l2_reg': float(self.l2_reg), 's_ll_reg': float(self.s_ll_reg), 'orth_reg': float(self.orth_reg)}


class SimilarityEncoder(object):

    def __init__(self, in_dim, embedding_dim, out_dim, hidden_layers=[], sparse_inputs=False, mask_value=None,
                 l2_reg=0.00000001, l2_reg_emb=0.00001, l2_reg_out=0., s_ll_reg=0., S_ll=None, orth_reg=0.,
                 opt=keras.optimizers.Adamax(lr=0.0005)):
        """
        Similarity Encoder (SimEc) neural network model

        Input:
            - in_dim: dimensionality of the input feature vector
            - embedding_dim: dimensionality of the embedding layer
            - out_dim: dimensionality of the output / number of targets
            - hidden_layers: list with tuples of (number of hidden units [int], activation function [string or keras function])
            - sparse_inputs: boolean, whether the input matrix is sparse (default: False)
            - mask_value: if some entries of the target matrix are missing, set them e.g. to -100 and then set
                          mask_value=-100 such that these entries are ignored when the backprop error is computed
            - l2_reg: float, l2 regularization strength of the hidden layers (default: 0.00000001)
            - l2_reg_emb: float, l2 regularization strength of the embedding (i.e. second to last) layer (default: 0.00001)
            - l2_reg_out: float, l2 regularization strength of the output (i.e. last) layer (default: 0.)
            - s_ll_reg: float, regularization strength for (S - W_-1^T W_-1), i.e. how much the dot product of the
                        last layer weights should approximate the target similarities; useful when factoring a square symmetric
                        similarity matrix. (default: 0.; if > 0. need to give S_ll)
            - S_ll: matrix that the dot product of the last layer should approximate (see above), needs to be (out_dim x out_dim)
            - orth_reg: float, regularization strength for (I - W_-1 W_-1^T), i.e. to encourage orthogonal rows in the last layer
            - opt: default: keras.optimizers.Adamax(lr=0.0005), the optimizer used for training the model
        """
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
        outputs = Dense(out_dim, activation='linear', use_bias=False,
                        kernel_regularizer=LastLayerReg(l2_reg_out, s_ll_reg, S_ll, orth_reg, embedding_dim, mask_value))(embedding)
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
        """
        Train the SimEc model

        Input:
            - X: n x in_dim feature matrix
            - S: n x out_dim target similarity matrix
            - epochs: int, number of epochs to train (default: 25)
            - verbose: given to the keras fit function, default: 1

        After training is complete, the SimEc object has another attribute "model_embed",
        which can be use to project the input feature vectors to the embedding space
        """
        assert X.shape[1] == self.in_dim, "input dim of data doesn't match (%i != %i)" % (X.shape[1], self.in_dim)
        assert X.shape[0] == S.shape[0], "number of samples for inputs and targets doesn't match (%i != %i)" % (X.shape[0], S.shape[0])
        assert S.shape[1] == self.out_dim, "output dim of targets doesn't match (%i != %i)" % (S.shape[1], self.out_dim)
        self.model.fit(X, S, epochs=epochs, verbose=verbose)
        # store the model we need for the prediction
        self.model_embed = Sequential(self.model.layers[:-1])

    def transform(self, X):
        """
        Project the input feature vectors to the embedding space

        Input:
            - X: m x in_dim feature matrix

        Returns:
            - Y: m x embedding_dim embedding matrix
        """
        assert self.model_embed is not None, "need to fit model first"
        return self.model_embed.predict(X)

    def predict(self, X):
        """
        Generate the output of the network, i.e. the predicted similarities

        Input:
            - X: m x in_dim feature matrix

        Returns:
            - S': m x out_dim output matrix with approximated similarities to the out_dim targets
        """
        assert self.model_embed is not None, "need to fit model first"
        return self.model_embed.predict(X)