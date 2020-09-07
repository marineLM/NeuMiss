'''Implements Neumann with the posibility to do batch learning'''

import math
import numpy as np
from sklearn.base import BaseEstimator

import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorchtools import EarlyStopping


class Neumann(nn.Module):
    def __init__(self, n_features, depth, residual_connection,  mlp_depth,
                 init_type):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.relu = nn.ReLU()

        # Create the parameters of the network
        l_W = [torch.empty(n_features, n_features, dtype=torch.float)
               for _ in range(self.depth)]
        Wc = torch.empty(n_features, n_features, dtype=torch.float)
        beta = torch.empty(1*n_features, dtype=torch.float)
        mu = torch.empty(n_features, dtype=torch.float)
        b = torch.empty(1, dtype=torch.float)
        l_W_mlp = [torch.empty(n_features, 1*n_features, dtype=torch.float)
                   for _ in range(mlp_depth)]
        l_b_mlp = [torch.empty(1*n_features, dtype=torch.float)
                   for _ in range(mlp_depth)]

        # Initialize the parameters of the network
        if init_type == 'normal':
            for W in l_W:
                nn.init.xavier_normal_(W)
            nn.init.xavier_normal_(Wc)
            nn.init.normal_(beta)
            nn.init.normal_(mu)
            nn.init.normal_(b)
            for W in l_W_mlp:
                nn.init.xavier_normal_(W)
            for b_mlp in l_b_mlp:
                nn.init.normal_(b_mlp)

        elif init_type == 'uniform':
            bound = 1 / math.sqrt(n_features)
            for W in l_W:
                nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            nn.init.kaiming_uniform_(Wc, a=math.sqrt(5))
            nn.init.uniform_(beta, -bound, bound)
            nn.init.uniform_(mu, -bound, bound)
            nn.init.normal_(b)
            for W in l_W_mlp:
                nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            for b_mlp in l_b_mlp:
                nn.init.uniform_(b_mlp, -bound, bound)

        # Make tensors learnable parameters
        self.l_W = [torch.nn.Parameter(W) for W in l_W]
        for i, W in enumerate(self.l_W):
            self.register_parameter('W_{}'.format(i), W)
        self.Wc = torch.nn.Parameter(Wc)
        self.beta = torch.nn.Parameter(beta)
        self.mu = torch.nn.Parameter(mu)
        self.b = torch.nn.Parameter(b)
        self.l_W_mlp = [torch.nn.Parameter(W) for W in l_W_mlp]
        for i, W in enumerate(self.l_W_mlp):
            self.register_parameter('W_mlp_{}'.format(i), W)
        self.l_b_mlp = [torch.nn.Parameter(b) for b in l_b_mlp]
        for i, b in enumerate(self.l_b_mlp):
            self.register_parameter('b_mlp_{}'.format(i), b)

    def forward(self, x, m, phase='train'):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, n_features)
            The input data imputed by 0.
        m: tensor, shape (batch_size, n_features)
            The missingness indicator (0 if observed and 1 if missing).
        """

        h0 = x + m*self.mu
        h = x - (1-m)*self.mu
        h_res = x - (1-m)*self.mu

        if len(self.l_W) > 0:
            S0 = self.l_W[0]
            h = torch.matmul(h, S0)*(1-m)

        for W in self.l_W[1:self.depth]:
            h = torch.matmul(h, W)*(1-m)
            if self.residual_connection:
                h += h_res

        h = torch.matmul(h, self.Wc)*m + h0
        if self.mlp_depth > 0:
            for W, b in zip(self.l_W_mlp, self.l_b_mlp):
                h = torch.matmul(h, W) + b
                h = self.relu(h)

        y = torch.matmul(h, self.beta)

        y = y + self.b

        return y


class Neumann_mlp(BaseEstimator):
    """The Neumann neural network

    Parameters
    ----------
    depth: int
        The number of Neumann iterations. Note that the total depth of the
        Neumann network will be `depth`+1 because of W_{mix}.

    n_epochs: int
        The maximum number of epochs.

    batch_size: int

    lr: float
        The learning rate.

    early_stopping: boolean
        If True, early stopping is used based on the validaton set, with a
        patience of 15 epochs.

    residual_connection: boolean
        If True, the residual connection of the Neumann network are
        implemented.

    mlp_depth: int
        The depth of the MLP stacked on top of the Neuman iterations.

    init_type: str
        The type of initialisation for the parameters. Either 'normal' or
        'uniform'.

    verbose: boolean
    """

    def __init__(self, depth, n_epochs, batch_size, lr,  early_stopping=False,
                 residual_connection=False, mlp_depth=0, init_type='normal',
                 verbose=False):
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stop = early_stopping
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.init_type = init_type
        self.verbose = verbose

        self.r2_train = []
        self.mse_train = []
        self.r2_val = []
        self.mse_val = []

    def fit(self, X, y, X_val=None, y_val=None):

        M = np.isnan(X)
        X = np.nan_to_num(X)

        n_samples, n_features = X.shape

        if X_val is not None:
            M_val = np.isnan(X_val)
            X_val = np.nan_to_num(X_val)

        M = torch.as_tensor(M, dtype=torch.float)
        X = torch.as_tensor(X, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float)

        if X_val is not None:
            M_val = torch.as_tensor(M_val, dtype=torch.float)
            X_val = torch.as_tensor(X_val, dtype=torch.float)
            y_val = torch.as_tensor(y_val, dtype=torch.float)

        self.net = Neumann(n_features=n_features, depth=self.depth,
                           residual_connection=self.residual_connection,
                           mlp_depth=self.mlp_depth, init_type=self.init_type)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
                        self.optimizer, mode='min', factor=0.2, patience=2,
                        threshold=1e-4)

        if self.early_stop and X_val is not None:
            early_stopping = EarlyStopping(verbose=self.verbose)

        running_loss = np.inf
        criterion = nn.MSELoss()

        # Train the network
        for i_epoch in range(self.n_epochs):
            if self.verbose:
                print("epoch nb {}".format(i_epoch))

            # Shuffle tensors to have different batches at each epoch
            ind = torch.randperm(n_samples)
            X = X[ind]
            M = M[ind]
            y = y[ind]

            xx = torch.split(X, split_size_or_sections=self.batch_size, dim=0)
            mm = torch.split(M, split_size_or_sections=self.batch_size, dim=0)
            yy = torch.split(y, split_size_or_sections=self.batch_size, dim=0)

            self.scheduler.step(running_loss/len(xx))

            param_group = self.optimizer.param_groups[0]
            lr = param_group['lr']
            if self.verbose:
                print("Current learning rate is: {}".format(lr))
            if lr < 5e-6:
                break

            running_loss = 0

            for bx, bm, by in zip(xx, mm, yy):

                self.optimizer.zero_grad()

                y_hat = self.net(bx, bm)

                loss = criterion(y_hat, by)
                running_loss += loss.item()
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                y_hat = self.net(X, M, phase='test')
                loss = criterion(y_hat, y)
                mse = loss.item()
                self.mse_train.append(mse)

                var = ((y - y.mean())**2).mean()
                r2 = 1 - mse/var
                self.r2_train.append(r2)

                if self.verbose:
                    print("Train loss - r2: {}, mse: {}".format(r2,
                          running_loss/len(xx)))

            # Evaluate the validation loss
            if X_val is not None:
                with torch.no_grad():
                    y_hat = self.net(X_val, M_val, phase='test')
                    loss_val = criterion(y_hat, y_val)
                    mse_val = loss_val.item()
                    self.mse_val.append(mse_val)

                    var = ((y_val - y_val.mean())**2).mean()
                    r2_val = 1 - mse_val/var
                    self.r2_val.append(r2_val)
                    if self.verbose:
                        print("Validation loss is: {}".format(r2_val))

                if self.early_stop:
                    early_stopping(mse_val, self.net)
                    if early_stopping.early_stop:
                        if self.verbose:
                            print("Early stopping")
                        break

        # load the last checkpoint with the best model
        if self.early_stop and early_stopping.early_stop:
            self.net.load_state_dict(early_stopping.checkpoint)

    def predict(self, X):

        M = np.isnan(X)
        X = np.nan_to_num(X)

        M = torch.as_tensor(M, dtype=torch.float)
        X = torch.as_tensor(X, dtype=torch.float)

        with torch.no_grad():
            y_hat = self.net(X, M, phase='test')

        return np.array(y_hat)
