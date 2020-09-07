import pandas as pd
import argparse
from estimators import ConstantImputedLR,\
                       EMLR,\
                       MICELR
from mlp import MLP_reg
from neumannS0_mlp import Neumann_mlp
from learning_curves import run

parser = argparse.ArgumentParser()
parser.add_argument('data_type', help='type of simulation',
                    choices=['gaussian_sm', 'probit_sm', 'MCAR',
                             'MAR_logistic'])
args = parser.parse_args()


n_iter = 20
n_jobs = 40
n_sizes = [2e4, 1e5]
n_sizes = [int(i) for i in n_sizes]
n_test = int(1e4)
n_val = int(1e4)

# First fill in data_desc with all default values.
if args.data_type == 'gaussian_sm':
    data_type = 'selfmasking'
    filename = 'gaussian_sm'
    compute_br = True
    default_values = {'n_features': 10, 'missing_rate': 0.5,
                      'prop_latent': 0.5, 'sm_type': 'gaussian', 'sm_param': 2,
                      'snr': 10, 'perm': False}

elif args.data_type == 'probit_sm':
    data_type = 'selfmasking'
    filename = 'probit_sm'
    compute_br = False
    default_values = {'n_features': 10, 'missing_rate': 0.5,
                      'prop_latent': 0.5, 'sm_type': 'probit', 'sm_param': 0.5,
                      'snr': 10, 'perm': False}

elif args.data_type == 'MCAR':
    data_type = 'MCAR'
    filename = 'MCAR'
    compute_br = True
    default_values = {'n_features': 10, 'missing_rate': 0.5,
                      'prop_latent': 0.5, 'snr': 10,
                      'masking': 'MCAR'}

elif args.data_type == 'MAR_logistic':
    data_type = 'MAR_logistic'
    filename = 'MAR_logistic'
    compute_br = True
    default_values = {'n_features': 10, 'missing_rate': 0.5,
                      'prop_latent': 0.5, 'snr': 10,
                      'masking': 'MAR_logistic', 'prop_for_masking': 0.1}



# Define the list of parameters that should be tested and their range of values
other_values = {'n_features': [20, 50]}

# Then vary parameters one by one while the other parameters remain constant,
# and equal to their default values.
data_descs = [pd.DataFrame([default_values])]
for param, vals in other_values.items():
    n = len(vals)
    data = pd.DataFrame([default_values]*n)
    data.loc[:, param] = vals
    data_descs.append(data)

data_descs = pd.concat(data_descs, axis=0)

methods = []
methods.append({'name': 'ConstantImputedLR', 'est': ConstantImputedLR})
methods.append({'name': 'EMLR', 'est': EMLR})
methods.append({'name': 'MICELR', 'est': MICELR})

for q in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]:
    methods.append({'name': 'torchMLP', 'est': MLP_reg, 'type_width': 'linear',
                    'width': q, 'depth': 1, 'n_epochs': 2000,
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
