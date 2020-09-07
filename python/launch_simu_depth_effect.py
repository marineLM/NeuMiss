import pandas as pd
from mlp import MLP_reg
from neumannS0_mlp import Neumann_mlp
from learning_curves import run


n_iter = 20
n_jobs = 40
n_sizes = [1e5]
n_sizes = [int(i) for i in n_sizes]
n_test = int(1e4)
n_val = int(1e4)
data_type = 'MCAR'
filename = 'MCAR_depth_effect'
compute_br = True

# First fill in data_desc with all default values.
default_values = {'n_features': 20, 'missing_rate': 0.5, 'prop_latent': 0.5,
                  'snr': 10, 'masking': 'MCAR'}
data_descs = pd.DataFrame([default_values])

methods = []

for q in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    methods.append({'name': 'torchMLP', 'est': MLP_reg, 'type_width': 'linear',
                    'width': q, 'depth': 1, 'n_epochs': 2000,
                    'batch_size': 200, 'early_stopping': True,
                    'verbose': False})

for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    methods.append({'name': 'torchMLP', 'est': MLP_reg, 'type_width': 'linear',
                    'width': 1, 'depth': d, 'n_epochs': 2000,
                    'batch_size': 200, 'early_stopping': True,
                    'verbose': False})

for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for residual_connection in [False, True]:
        for early_stopping in [False, True]:
            methods.append(
                {'name': 'Neumann', 'est': Neumann_mlp, 'depth': d,
                 'n_epochs': 100, 'batch_size': 10,
                 'early_stopping': early_stopping,
                 'residual_connection': residual_connection,
                 'verbose': False})

run_params = {
        'n_iter': n_iter,
        'n_sizes': n_sizes,
        'n_test': n_test,
        'n_val': n_val,
        'data_type': data_type,
        'data_descs': data_descs,
        'methods': methods,
        'compute_br': compute_br,
        'filename': filename,
        'n_jobs': n_jobs}

run(**run_params)
