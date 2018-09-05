from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range, object
from copy import deepcopy
import random
random.seed(28)
import numpy as np
np.random.seed(28)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
torch.manual_seed(28)
torch.cuda.manual_seed(28)
torch.backends.cudnn.deterministic = True


class Dense(nn.Linear):
    """
    Shorthand for a nn.Linear layer with an activation function

    Args:
        in_dim (int): number of input feature
        out_dim (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function or string (default: None)
    """
    def __init__(self, in_dim, out_dim, bias=True, activation=None):
        activation_map = {"tanh": torch.tanh,
                          "sigmoid": torch.sigmoid,
                          "relu": torch.relu}
        if activation in activation_map:
            activation = activation_map[activation]
        self.activation = activation
        super(Dense, self).__init__(in_dim, out_dim, bias)

    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)
        return y


class FFNet(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_layers=[]):
        """
        Neural network PyTorch model; shortcut for creating a feed forward NN that can be used as an in_net for a SimEcModel

        Input:
            - in_dim: input dimensionality
            - out_dim: output dimensionality
            - hidden_layers: list with tuples of (number of hidden units [int], activation function [string or function])
        """
        super(FFNet, self).__init__()
        # get a list of layer dimensions: in_dim --> hl --> out_dim
        dimensions = [in_dim]
        dimensions.extend([h[0] for h in hidden_layers])
        dimensions.append(out_dim)
        # get list of activation functions (output (i.e. embedding) layer has no activation)
        activations = [h[1] for h in hidden_layers]
        activations.append(None)
        # initialize dense layers
        layers = [Dense(dimensions[i], dimensions[i+1], activation=activations[i]) for i in range(len(activations))]
        # construct feed forward network
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class SimEcModel(nn.Module):

    def __init__(self, in_net, embedding_dim, out_dim):
        """
        Similarity Encoder (SimEc) neural network PyTorch model

        Input:
            - in_net: input network mapping from whatever original input to the embedding (e.g. a FFNet)
            - embedding_dim: dimensionality of the embedding layer
            - out_dim: dimensionality of the output / number of targets
        """
        super(SimEcModel, self).__init__()
        # the simec model is the in_net, which creates the embedding,
        self.embedding_net = in_net
        # plus a last layer to compute the similarity approximation
        self.W_ll = Dense(embedding_dim, out_dim, bias=False)

    def forward(self, inputs):
        x = self.embedding_net(inputs)
        x = self.W_ll(x)
        return x


class SimilarityEncoder(object):

    def __init__(self, in_net, embedding_dim, out_dim, **kwargs):
        """
        Similarity Encoder (SimEc) neural network model wrapper

        Input:
            - in_net: either a NN model that maps from the input to the embedding OR the dimensionality of
                      the input feature vector (int), in which case a FFNet will be created by supplying
                      the kwargs, which should probably contain a "hidden_layers" argument
            - embedding_dim: dimensionality of the embedding layer
            - out_dim: dimensionality of the output / number of targets
        """
        if isinstance(in_net, int):
            in_net = FFNet(in_net, embedding_dim, **kwargs)
        self.model = SimEcModel(in_net, embedding_dim, out_dim)
        self.device = "cpu"  # by default, before training, the model is on the cpu

    def fit(self, X, S, epochs=25, batch_size=32, lr=0.0005, weight_decay=0., s_ll_reg=0., S_ll=None, orth_reg=0.):
        """
        Train the SimEc model

        Input:
            - X: n x in_dim feature matrix
            - S: n x out_dim target similarity matrix
            - epochs: int, number of epochs to train (default: 25)
            - batch_size: int, number of samples per batch (default: 32)
            - lr: float used as the learning rate for the Adam optimizer (default: lr=0.0005)
            - weight_decay: l2 regularization, given as a parameter to the optimizer
            - s_ll_reg: float, regularization strength for (S - W_ll^T W_ll), i.e. how much the dot product of the
                        last layer weights should approximate the target similarities; useful when factoring a square symmetric
                        similarity matrix. (default: 0.; if > 0. need to give S_ll)
            - S_ll: matrix that the dot product of the last layer should approximate (see above), needs to be (out_dim x out_dim)
            - orth_reg: float, regularization strength for (lambda*I - W_ll W_ll^T), i.e. to encourage orthogonal rows in the last layer
                        usually only helpful when using many embedding dimensions (> 100)
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()

        if s_ll_reg > 0:
            if S_ll is None:
                print("Warning: need to give S_ll if s_ll_reg > 0.")
                s_ll_reg = 0.
            else:
                S_ll = torch.from_numpy(S_ll).float()
                S_ll = S_ll.to(self.device)
        if orth_reg > 0:
            edim = self.model.W_ll.weight.size()[1]
            Ones = torch.from_numpy((np.ones((edim, edim)) - np.eye(edim))).float()
            Ones = Ones.to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)

        kwargs = {'num_workers': 1, 'pin_memory': True} if not self.device == "cpu" else {}
        trainloader = tdata.DataLoader(tdata.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(S).float()),
                                       batch_size=batch_size, shuffle=False, **kwargs)
        # loop over the dataset multiple times
        best_loss = np.inf
        best_model = None
        for epoch in range(epochs):

            running_loss = 0.0
            for i, data in enumerate(trainloader):
                # get the inputs
                x_batch, s_batch = data
                x_batch, s_batch = x_batch.to(self.device), s_batch.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(x_batch)
                loss = criterion(outputs, s_batch)
                running_loss += loss.item()
                # possibly some regularization stuff based on the last layer
                # CAREFUL! linear layer weights are stored with dimensions (out_dim, in_dim) instead of
                # how you initialized them with (in_dim, out_dim), therefore the transpose is mixed up here!
                if s_ll_reg > 0:
                    loss += s_ll_reg*criterion(torch.mm(self.model.W_ll.weight, self.model.W_ll.weight.t()), S_ll)
                if orth_reg > 0:
                    loss += orth_reg * torch.mean((Ones * torch.mm(self.model.W_ll.weight.t(), self.model.W_ll.weight))**2)
                loss.backward()
                optimizer.step()
            if epoch > 10:
                lr_scheduler.step(running_loss)
            print('[epoch %d] loss: %.7f' % (epoch + 1, running_loss / (i + 1)))
            # in case the learning rate was too high or something we keep track
            # of the model with the lowest error and use that in the end
            if running_loss < best_loss:
                best_loss = running_loss
                best_model = deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_model)

    def transform(self, X):
        """
        Project the input feature vectors to the embedding space

        Input:
            - X: m x in_dim feature matrix

        Returns:
            - Y: m x embedding_dim embedding matrix
        """
        self.model.eval()
        X = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            Y = self.model.embedding_net(X).cpu()
        return Y.numpy()

    def predict(self, X):
        """
        Generate the output of the network, i.e. the predicted similarities

        Input:
            - X: m x in_dim feature matrix

        Returns:
            - S': m x out_dim output matrix with approximated similarities to the out_dim targets
        """
        self.model.eval()
        X = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            S = self.model(X).cpu()
        return S.numpy()
