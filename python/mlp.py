import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorchtools import EarlyStopping


class Mlp(nn.Module):

    def __init__(self, n_features, hidden_layer_sizes):
        super(Mlp, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.relu = nn.ReLU()

        # Architecture
        self.layers = nn.ModuleList()  # list weights matrix
        self.layers.append(nn.Linear(n_features, hidden_layer_sizes[0]))

        for d_in, d_out in zip(hidden_layer_sizes,
                               hidden_layer_sizes[1:] + [1]):
            self.layers.append(nn.Linear(d_in, d_out))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
        x = self.layers[-1](x).view(-1)
        return x


class MLP_reg(RegressorMixin):

    def __init__(self, hidden_layer_sizes=[], lr=.001, batch_size=10,
                 n_epochs=100, impute_strategy='constant',
                 early_stopping=False, verbose=False):

        if hidden_layer_sizes == []:
            hidden_layer_sizes = [500]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stop = early_stopping
        self.verbose = verbose

        self.i_epoch = 0
        self._scaler = StandardScaler()
        self._imputer = SimpleImputer(strategy=impute_strategy, fill_value=0)
        self.criterion = nn.MSELoss()

        # Save performances
        self.mse_train = []
        self.mse_val = []
        self.r2_train = []
        self.r2_val = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        n_samples, n_features = X_train.shape

        # Add mask, impute and scale
        X_train = self._add_mask(X_train)
        X_train = self._imputer.fit_transform(X_train)
        X_train = self._scaler.fit_transform(X_train)
        X_train = torch.as_tensor(X_train, dtype=torch.float)
        y_train = torch.as_tensor(y_train, dtype=torch.float)

        if X_val is not None:
            X_val = self._add_mask(X_val)
            X_val = self._imputer.transform(X_val)
            X_val = self._scaler.transform(X_val)
            X_val = torch.as_tensor(X_val, dtype=torch.float)
            y_val = torch.as_tensor(y_val, dtype=torch.float)

        self.net = Mlp(n_features=2*n_features,
                       hidden_layer_sizes=self.hidden_layer_sizes)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                           factor=0.2, patience=2,
                                           threshold=1e-4)

        if self.early_stop and X_val is not None:
            early_stopping = EarlyStopping(verbose=self.verbose)
        running_loss = np.inf

        for _ in range(self.n_epochs):
            # Run one eopch
            self.i_epoch += 1

            ind = torch.randperm(n_samples)
            X_train = X_train[ind]
            y_train = y_train[ind]

            xx = torch.split(
                X_train, split_size_or_sections=self.batch_size, dim=0)
            yy = torch.split(
                y_train, split_size_or_sections=self.batch_size, dim=0)

            self.scheduler.step(running_loss/len(xx))

            param_group = self.optimizer.param_groups[0]
            lr = param_group['lr']
            if self.i_epoch % 10 == 0 and self.verbose:
                print("Current learning rate is: {}".format(lr))
            if lr < 5e-6:
                break

            running_loss = 0

            for bx, by in zip(xx, yy):

                self.optimizer.zero_grad()

                y_hat = self.net(bx)

                loss = self.criterion(y_hat, by)
                running_loss += loss.item()
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            mse, r2 = self.get_score(X_train, y_train)
            self.mse_train.append(mse)
            self.r2_train.append(r2)

            # Print epoch info
            if self.i_epoch % 10 == 0 and self.verbose:
                print("epoch nb {}".format(self.i_epoch))
                print("Train loss is: {}".format(r2))

            # Evaluate validation loss
            if X_val is not None:
                mse_val, r2_val = self.get_score(X_val, y_val)
                self.mse_val.append(mse_val)
                self.r2_val.append(r2_val)
                if self.i_epoch % 10 == 0 and self.verbose:
                    print("Validation loss is: {}".format(r2_val))

                if self.early_stop:
                    early_stopping(mse_val, self.net)
                    if early_stopping.early_stop:
                        if self.verbose:
                            print("Early stopping")
                        break

        # load the last checkpoint with the best model
        if self.early_stop and X_val is not None:
            self.net.load_state_dict(early_stopping.checkpoint)

        print(self.hidden_layer_sizes, self.n_epochs, 'fitted')
        return self

    def predict(self, X, return_numpy=True):
        T = self._add_mask(X)
        T = self._imputer.transform(T)
        T = self._scaler.transform(T)
        T = torch.as_tensor(T, dtype=torch.float)

        with torch.no_grad():
            ans = self.net(T).detach()

        if return_numpy:
            return ans.numpy()
        return ans

    def _add_mask(self, X):
        M = np.isnan(X)
        return np.hstack((X, M))

    def get_score(self, X, y):
        with torch.no_grad():
            y_hat = self.net(X)
            mse = self.criterion(y_hat, y)
            var = ((y - y.mean())**2).mean()
            r2 = 1 - mse/var
        return mse, r2
