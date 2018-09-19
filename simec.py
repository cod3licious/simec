from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range, object
import numpy as np
np.random.seed(28)
import scipy.sparse as sp
import tensorflow as tf
tf.set_random_seed(28)
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape
from keras.regularizers import Regularizer
from keras.losses import mean_squared_error, binary_crossentropy


def generate_from_sparse_targets(X, S, mask_value, batch_size=32, shuffle=True):
    # save the shape here in case S is a tensor
    S_out_shape = list(S.shape)
    while True:
        # get index for every epoch
        if shuffle:
            idx = np.array(np.random.permutation(X.shape[0]), dtype=np.int16)
        else:
            idx = np.arange(X.shape[0], dtype=np.int16)
        # generate data for each batch
        for i in range(int(np.ceil(X.shape[0]/batch_size))):
            b_idx = idx[i*batch_size:(i+1)*batch_size]
            S_out_shape[0] = len(b_idx)
            # missing values will be mask_value
            S_batch = mask_value*np.ones(S_out_shape, dtype=np.float16)
            # other entries are filled with entries from corresponding rows of S
            S_batch[S[b_idx].nonzero()] = S[b_idx][S[b_idx].nonzero()]
            yield X[b_idx], S_batch


def masked_mse(mask_value):
    """
    https://github.com/fchollet/keras/issues/7065
    compute mean squared error using only those values not equal to mask_value,
    e.g. to deal with missing values in the target similarity matrix
    """
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        # in case mask_true is 0 everywhere, the error would be nan, therefore divide by at least 1
        # this doesn't change anything as where sum(mask_true)==0, sum(masked_squared_error)==0 as well
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_mse
    f.__name__ = str('Masked MSE (mask_value={})'.format(mask_value))
    return f


def masked_binary_crossentropy(mask_value):
    """
    compute binary cross-entropy using only those values not equal to mask_value,
    e.g. to deal with missing values in the target similarity matrix
    """
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_bce = mask_true * binary_crossentropy(y_true, y_pred)
        # in case mask_true is 0 everywhere, the error would be nan, therefore divide by at least 1
        # this doesn't change anything as where sum(mask_true)==0, sum(masked_bce)==0 as well
        masked_bce = K.sum(masked_bce, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_bce
    f.__name__ = str('Masked Binary Cross-Entropy (mask_value={})'.format(mask_value))
    return f


class LastLayerReg(Regularizer):

    def __init__(self, l2_reg=0., s_ll_reg=0., S_ll=None, orth_reg=0., embedding_dim=0, reshape=None, mask_value=None):
        """
        Custom regularizer used for the last layer of a SimEc
        s_ll_reg enforces that W^TW approximates S,
        orth_reg enforces that WW^T approximates lambda*I, i.e. that the vectors are orthogonal (but not necessarily length 1)
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
        self.reshape = reshape
        if mask_value is None:
            self.errfun = mean_squared_error
        else:
            self.errfun = masked_mse(mask_value)

    def __call__(self, x):
        regularization = 0.
        if self.l2_reg > 0.:
            regularization += K.sum(self.l2_reg * K.square(x))
        if self.reshape is None:
            if self.s_ll_reg > 0.:
                regularization += self.s_ll_reg * K.mean(self.errfun(self.S_ll, K.dot(K.transpose(x), x)))
            if self.orth_reg > 0.:
                regularization += self.orth_reg * K.mean(K.square((K.ones((self.embedding_dim, self.embedding_dim)) - K.eye(self.embedding_dim)) * K.dot(x, K.transpose(x))))
        else:
            x_reshaped = K.reshape(x, self.reshape)
            for i in range(self.reshape[2]):
                if self.s_ll_reg > 0.:
                    regularization += self.s_ll_reg * K.mean(self.errfun(self.S_ll[:,:,i], K.dot(K.transpose(x_reshaped[:,:,i]), x_reshaped[:,:,i])))
                if self.orth_reg > 0.:
                    regularization += self.orth_reg * K.mean(K.square((K.ones((self.embedding_dim, self.embedding_dim)) - K.eye(self.embedding_dim)) * K.dot(x_reshaped[:,:,i], K.transpose(x_reshaped[:,:,i]))))
        return regularization

    def get_config(self):
        return {'l2_reg': float(self.l2_reg), 's_ll_reg': float(self.s_ll_reg), 'orth_reg': float(self.orth_reg)}


class SimilarityEncoder(object):

    def __init__(self, in_dim, embedding_dim, out_dim, hidden_layers=[], sparse_inputs=False, mask_value=None,
                 l2_reg=0.00000001, l2_reg_emb=0.00001, l2_reg_out=0., s_ll_reg=0., S_ll=None, orth_reg=0., W_ll=None,
                 opt=0.0005, loss='mse', ll_activation='linear'):
        """
        Similarity Encoder (SimEc) neural network model

        Input:
            - in_dim: dimensionality of the input feature vector
            - embedding_dim: dimensionality of the embedding layer
            - out_dim: dimensionality of the output / number of targets; if out_dim is a tuple, e.g. (n_targets, n_similarities)
                       then s_ll_reg and orth_reg are ignored
            - hidden_layers: list with tuples of (number of hidden units [int], activation function [string or keras function])
            - sparse_inputs: boolean, whether the input matrix is sparse (default: False)
            - mask_value: if some entries of the target matrix are missing, set them e.g. to -100 and then set
                          mask_value=-100 such that these entries are ignored when the backprop error is computed
            - l2_reg: float, l2 regularization strength of the hidden layers (default: 0.00000001)
            - l2_reg_emb: float, l2 regularization strength of the embedding (i.e. second to last) layer (default: 0.00001)
            - l2_reg_out: float, l2 regularization strength of the output (i.e. last) layer (default: 0.)
            - s_ll_reg: float, regularization strength for (S - W_ll^T W_ll), i.e. how much the dot product of the
                        last layer weights should approximate the target similarities; useful when factoring a square symmetric
                        similarity matrix. (default: 0.; if > 0. need to give S_ll)
            - S_ll: matrix that the dot product of the last layer should approximate (see above), needs to be (out_dim x out_dim)
            - orth_reg: float, regularization strength for (lambda*I - W_ll W_ll^T), i.e. to encourage orthogonal rows in the last layer
                        usually only helpful when using many embedding dimensions (> 100)
            - W_ll: matrix that should be used as the frozen weights of the last layer; this should be used if you factorize
                    an (m x n) matrix R and want to get the mapping for both some (m x D) features as well as some (n x P) features.
                    To do this, first train a SimEc to approximate R using the (m x D) feature matrix as input. After training,
                    use simec.transform(X) to get the (m x embedding_dim) embedding Y. Then train another SimEc using the
                    (n x P) feature matrix as input to approximate R.T and this time set W_ll=Y.T. Then, with both SimEcs you
                    can project the (m x D) as well as the (n x P) feature vectors into the same embedding space where their
                    scalar product approximates R.
            - opt: either a float used as the learning rate for keras.optimizers.Adamax (default: lr=0.0005),
                   or a keras optimizers instance that should be used for training the model
            - loss: which loss function to use (if mask_value != None, only 'mse' or 'binary_crossentropy'; default: loss='mse').
            - ll_activation: activation function on the last layer. If a different loss than mse is used,
                             this should probably be changed as well (default: 'linear').
        """
        # save some parameters we might need for later checks
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reshape_output = None
        self.mask_value = mask_value
        ll_reshape = None
        if isinstance(out_dim, tuple):
            ll_reshape = (embedding_dim, out_dim[0], out_dim[1])
            out_dim, self.reshape_output = np.prod(out_dim), out_dim
        # checks for s_ll_regularization
        if s_ll_reg > 0.:
            assert S_ll is not None, "need S_ll"
            if self.reshape_output is None:
                assert S_ll.shape == (out_dim, out_dim), "S_ll needs to be of shape (out_dim x out_dim)"
            else:
                assert S_ll.shape == (self.reshape_output[0], self.reshape_output[0], self.reshape_output[1]), "S_ll needs to be of shape (out_dim x out_dim x n_similarities)"
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
        if W_ll is None:
            outputs = Dense(out_dim, activation=ll_activation, use_bias=False,
                            kernel_regularizer=LastLayerReg(l2_reg_out, s_ll_reg, S_ll, orth_reg, embedding_dim, ll_reshape, mask_value))(embedding)
        else:
            assert W_ll.shape[0] == embedding_dim, "W_ll.shape[0] doesn't match embedding_dim (%i != %i)" % (W_ll.shape[0], embedding_dim)
            assert W_ll.shape[1] == out_dim, "W_ll.shape[1] doesn't match out_dim (%i != %i)" % (W_ll.shape[1], out_dim)
            outputs = Dense(out_dim, activation=ll_activation, use_bias=False, trainable=False, weights=[W_ll])(embedding)
        # possibly reshape the output if multiple similarities are used as targets
        if self.reshape_output is not None:
            outputs = Reshape(self.reshape_output)(outputs)
        # put it all into a model
        self.model = Model(inputs=inputs, outputs=outputs)
        # compile the model to minimize the MSE
        if isinstance(opt, float):
            opt = keras.optimizers.Adamax(lr=opt)
        if mask_value is None:
            self.model.compile(optimizer=opt, loss=loss)
        else:
            assert loss in ("mse", "binary_crossentropy"), "Loss %s not implemented for target matrices with missing values. Use 'mse' or 'binary_crossentropy'." % loss
            if loss == "binary_crossentropy":
                self.model.compile(optimizer=opt, loss=masked_binary_crossentropy(mask_value))
            else:
                self.model.compile(optimizer=opt, loss=masked_mse(mask_value))

        # placeholder for embedding model
        self.model_embed = None

    def fit(self, X, S, epochs=25, batch_size=32, verbose=1):
        """
        Train the SimEc model

        Input:
            - X: n x in_dim feature matrix
            - S: n x out_dim target similarity matrix
            - epochs: int, number of epochs to train (default: 25)
            - batch_size: int, number of samples per batch (default: 32)
            - verbose: given to the keras fit function, default: 1

        After training is complete, the SimEc object has another attribute "model_embed",
        which can be use to project the input feature vectors to the embedding space
        """
        if np.max(np.abs(S)) > 5.:
            print("Warning: For best results, S (and X) should be normalized (try S /= np.max(np.abs(S))).")
        assert X.shape[1] == self.in_dim, "input dim of data doesn't match (%i != %i)" % (X.shape[1], self.in_dim)
        assert X.shape[0] == S.shape[0], "number of samples for inputs and targets doesn't match (%i != %i)" % (X.shape[0], S.shape[0])
        if self.reshape_output is None:
            assert S.shape[1] == self.out_dim, "output dim of targets doesn't match (%i != %i)" % (S.shape[1], self.out_dim)
        else:
            assert S.shape[1:] == self.reshape_output, "output dims of targets don't match (%r != %r)" % (S.shape[1:], self.reshape_output)
        if self.mask_value is not None and sp.issparse(S):
            self.model.fit_generator(generate_from_sparse_targets(X, S, self.mask_value, batch_size),
                                     epochs=epochs, verbose=verbose, steps_per_epoch=int(np.ceil(X.shape[0]/batch_size)))
        else:
            self.model.fit(X, S, epochs=epochs, batch_size=batch_size, verbose=verbose)
        # store the model we need for the prediction
        if self.reshape_output is None:
            self.model_embed = Sequential(self.model.layers[:-1])
        else:
            self.model_embed = Sequential(self.model.layers[:-2])

    def transform(self, X, warn=True):
        """
        Project the input feature vectors to the embedding space

        Input:
            - X: m x in_dim feature matrix

        Returns:
            - Y: m x embedding_dim embedding matrix
        """
        assert X.shape[1] == self.in_dim, "input dim of data doesn't match (%i != %i)" % (X.shape[1], self.in_dim)
        if self.model_embed is None and warn:
            print("WARNING: model is not fitted yet!")
        return self.model_embed.predict(X)

    def predict(self, X, warn=True):
        """
        Generate the output of the network, i.e. the predicted similarities

        Input:
            - X: m x in_dim feature matrix

        Returns:
            - S': m x out_dim output matrix with approximated similarities to the out_dim targets
        """
        assert X.shape[1] == self.in_dim, "input dim of data doesn't match (%i != %i)" % (X.shape[1], self.in_dim)
        if self.model_embed is None and warn:
            print("WARNING: model is not fitted yet!")
        return self.model.predict(X)
