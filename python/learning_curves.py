import pandas as pd
import numpy as np
from collections import namedtuple
from ground_truth import gen_params, gen_data,\
                         gen_params_selfmasking, gen_data_selfmasking,\
                         BayesPredictor_gaussian_selfmasking,\
                         BayesPredictor_MCAR_MAR

from joblib import Memory, Parallel, delayed
location = './cachedir'
memory = Memory(location, verbose=0)

# Result item to create the DataFrame in a consistent way.
fields = ['key', 'method', 'train_test', 'n', 'mse', 'r2', 'early_stopping',
          'residual_connection', 'depth', 'n_epochs', 'lr', 'batch_size',
          'type_width', 'width', 'verbose', 'learning_rate',
          'learning_rate_init', 'max_iter', 'validation_fraction', 'mlp_depth']
# This only works starting from Python 3.7
# ResultItem = namedtuple('ResultItem', fields, defaults=(np.nan, )*len(fields))
# For older version of Python:
ResultItem = namedtuple('ResultItem', fields)
ResultItem.__new__.__defaults__ = (np.nan, )*len(ResultItem._fields)


@memory.cache
def run_one(X, y, est, params, method, n_test, n_val):
    n, p = X.shape
    n = n - n_val - n_test
    if method == 'torchMLP':
        print('method: {}, dim: {}, width: {}'.format(
            method, (n, p), params['hidden_layer_sizes'][0]))
    elif method == 'Neumann':
        print('method: {}, dim: {}, depth: {}, early_stop: {}, res: {}'.format(
            method, (n, p), params['depth'], params['early_stopping'],
            params['residual_connection']))
    else:
        print('method: {}, dim: {}'.format(method, (n, p)))

    X_test = X[0:n_test]
    y_test = y[0:n_test]
    X_val = X[n_test:(n_test + n_val)]
    y_val = y[n_test:(n_test + n_val)]
    X_train = X[(n_test + n_val):]
    y_train = y[(n_test + n_val):]

    if n_val > 0 and method in ['Neumann', 'torchMLP']:
        reg = est(**params)
        reg.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    elif method == 'MICEMLP':
        X_train = X[n_test:]
        y_train = y[n_test:]
        reg = est(params)
        reg.fit(X_train, y_train)
    else:
        reg = est(**params)
        reg.fit(X_train, y_train)

    pred_test = reg.predict(X_test)
    pred_train = reg.predict(X_train)
    pred_val = reg.predict(X_val)

    mse_train = ((y_train - pred_train)**2).mean()
    mse_test = ((y_test - pred_test)**2).mean()
    mse_val = ((y_val - pred_val)**2).mean()

    var_train = ((y_train - y_train.mean())**2).mean()
    var_test = ((y_test - y_test.mean())**2).mean()
    var_val = ((y_val - y_val.mean())**2).mean()

    r2_train = 1 - mse_train/var_train
    r2_test = 1 - mse_test/var_test
    r2_val = 1 - mse_val/var_val

    return {'train': {'mse': mse_train, 'r2': r2_train},
            'test': {'mse': mse_test, 'r2': r2_test},
            'val': {'mse': mse_val, 'r2': r2_val}
            }


def get_results(key, g, methods, n_test, n_val):

    result_iter = []

    gen = g['gen']
    data_params = g['data_params']

    for X, y in gen:
        n, p = X.shape

        for est_params in methods:

            params = est_params.copy()

            est = params.pop('est')
            method = params.pop('name')

            if method in ['torchMLP', 'Neumann']:
                params['lr'] = 1e-2/p

            if method == 'torchMLP':
                type_width = params.pop('type_width')
                q = params.pop('width')
                d = params.pop('depth')

                if type_width == 'exponential':
                    n_shallow = int(q*2**p)
                elif type_width == 'linear':
                    n_shallow = int(q*p)

                hidden_layer_sizes = [n_shallow]*d
                params['hidden_layer_sizes'] = hidden_layer_sizes

            if method == 'BayesPredictor':
                params['data_params'] = data_params

            new_score = run_one(X, y, est, params, method, n_test, n_val)

            if method == 'torchMLP':
                params = est_params.copy()
                params.pop('est')
                params.pop('name')

            if method == 'BayesPredictor':
                params.pop('data_params')

            res_train = ResultItem(key=key, method=method, train_test="train",
                                   n=n-n_test-n_val, **new_score["train"],
                                   **params)
            res_test = ResultItem(key=key, method=method, train_test="test",
                                  n=n-n_test-n_val, **new_score["test"],
                                  **params)
            res_val = ResultItem(key=key, method=method, train_test="val",
                                 n=n-n_test-n_val, **new_score["val"],
                                 **params)

            result_iter.extend([res_train, res_test, res_val])

    return result_iter


def run(n_iter, n_sizes, n_test, n_val, data_type, data_descs, methods,
        compute_br, filename, n_jobs=1):

    if data_type == 'selfmasking':
        generate_params = gen_params_selfmasking
        generate_data = gen_data_selfmasking
    elif data_type in ['MCAR', 'MAR_logistic']:
        generate_params = gen_params
        generate_data = gen_data

    if compute_br:
        if data_type in ['MCAR', 'MAR_logistic']:
            methods.append({'name': 'BayesPredictor',
                            'est': BayesPredictor_MCAR_MAR})

        elif (data_type in ['selfmasking'] and filename == 'gaussian_sm'):
            methods.append({'name': 'BayesPredictor',
                            'est': BayesPredictor_gaussian_selfmasking})
        else:
            raise ValueError('Bayes rate cannot be computed for' +
                             'data_type {}'.format(filename))

    # data_generators will contain the list of generators for each data
    # description, as well as the data parameters.
    data_generators = []
    key = -1
    for data_desc in data_descs.itertuples(index=False):
        data_desc = dict(data_desc._asdict())

        for it in range(n_iter):

            key += 1
            data_params = generate_params(**data_desc, random_state=it)

            # if compute_br:
            #     if data_type in ['MCAR', 'MAR_logistic']:
            #         methods.append(
            #             {'name': 'BayesPredictor',
            #              'est': BayesPredictor_MCAR_MAR,
            #              'data_params': data_param}
            #         )

            #     elif (data_type in ['selfmasking'] and
            #           data_param[1] == 'gaussian'):
            #         methods.append(
            #             {'name': 'BayesPredictor',
            #              'est': BayesPredictor_gaussian_selfmasking,
            #              'data_params': data_param}
            #         )
            #     else:
            #         raise ValueError('Bayes rate cannot be computed for' +
            #                          'data_type {}'.format(data_type))

            #     br_item = ResultItem(
            #         key=key, method='BayesRate', train_test='test', n=np.nan,
            #         mse=br['mse'], r2=br['r2'])
            #     bayes_rates.append(br_item)

            n_tot = [n_train + n_test + n_val for n_train in n_sizes]
            gen = generate_data(n_tot, data_params, random_state=it)
            data_generators.append(
                {'gen': list(gen), 'data_params': data_params})

    # Update data_descs so that the iteration ID is taken into account,
    # and add a key column to be able to merge the description dataframe with
    # the results dataframe.
    df_iter = pd.DataFrame({'iter': np.arange(n_iter)})
    data_descs = pd.merge(data_descs.assign(key=0), df_iter.assign(key=0),
                          on='key').drop('key', axis=1)
    data_descs = data_descs.assign(key=range(key+1))
    data_descs = data_descs.set_index('key')

    # Compute the results
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_results)(key, g, methods, n_test, n_val)
        for key, g in enumerate(data_generators)
    )
    results = [item for result_iter in results for item in result_iter]
    results = pd.DataFrame(results)
    results = results.set_index('key')

    # Merge the description and results dataframes
    scores = data_descs.join(results, how='outer')

    # if compute_br:
    #     results_br = pd.DataFrame(bayes_rates)
    #     results_br = results_br.set_index('key')
    #     scores_br = data_descs.join(results_br, how='outer')
    #     scores = pd.concat([scores_meth, scores_br], axis=0, sort=False)
    # else:
    #     scores = scores_meth

    scores.to_csv('../results/' + filename + '.csv')
