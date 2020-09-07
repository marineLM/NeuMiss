import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_one(ax, data, is_legend=False):

    if is_legend:
        is_legend = 'full'

    ax.axhline(0, color='.8', lw=2)

    # Plot train
    sns.lineplot(
        data=data.query('train_test == "train"'), x='depth_or_width',
        y='r2', hue='method', ax=ax, ci=None,
        legend=False, style="train_test", dashes=6 * [(1, 2)])

    # Plot test
    sns.lineplot(
        data=data.query('train_test == "test"'), x='depth_or_width',
        y='r2', hue='method', ax=ax, estimator=np.median, legend=is_legend)

    # Set axes
    ax.set_xlabel('depth or width')
    ax.set_ylabel('Bayes rate - {} score'.format('R2'))
    ax.grid(True)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 0.2)
    # axes[i].set_ylim(-0.05, 0.3)


if __name__ == '__main__':

    for data_type in ['MCAR']:

        scores = pd.read_csv('../results/' + data_type + '.csv',
                             index_col=0)

        # Separate Bayes rate from other methods performances
        if data_type in ['MCAR']:
            br = scores.query('method == "BayesRate"')
            scores = scores.query('method != "BayesRate"')

        # Differentiate Neumann and Neumann flex
        res = scores.residual_connection.fillna(False)
        es = scores.early_stopping.fillna(False)
        neu = (scores.method == 'Neumann')

        ind_res = (neu) & (res) & (~es)
        ind_es = (neu) & (~res) & (es)
        ind_res_es = (neu) & (res) & (es)

        scores.loc[ind_res, 'method'] = 'Neumann_res'
        scores.loc[ind_es, 'method'] = 'Neumann_es'
        scores.loc[ind_res_es, 'method'] = 'Neumann_res_es'

        # Choose the methods to be plotted
        methods = ['ConstantImputedLR', 'EMLR', 'MICELR', 'torchMLP',
                   'Neumann', 'Neumann_res', 'Neumann_es', 'Neumann_res_es']

        scores = scores.query('method in @methods')

        # Create a column depth_or_width that will give the depth of Neumann
        # or Schultz and the width of the MLP
        scores['depth_or_width'] = scores.width.copy()
        scores.depth_or_width.fillna(scores.depth, inplace=True)

        # Duplicate the rows corresponding to shallow methods so that they can
        # be plotted as lines
        shallow_methods = ['ConstantImputedLR', 'EMLR', 'MICELR']
        is_shallow = scores.method.isin(shallow_methods)
        scores_shallow = scores[is_shallow]
        scores_shallow.depth_or_width = scores.depth_or_width.max()
        scores.loc[is_shallow, 'depth_or_width'] = scores.depth_or_width.min()
        scores = pd.concat([scores, scores_shallow], axis=0)

        fig, axes = plt.subplots(3, 2, sharey=True, sharex=True,
                                 figsize=(10, 10))

        n_features = scores.n_features.unique()
        n_samples = scores.n.dropna().unique()

        for i, p in enumerate(n_features):
            row_axes = axes[i]
            data = scores.query('n_features == @p')

            # Compute the performances ajusted for the Bayes rate
            if data_type in ['MCAR']:
                for it in data.iter.unique():
                    br_it = br.loc[(br.iter == it) & (br.n_features == p), 'r2']
                    br_it = float(br_it)
                    data.loc[data.iter == it, 'r2'] = (
                        br_it - data.loc[data.iter == it, 'r2'])

            for j, n in enumerate(n_samples):
                data_n = data.query('n == @n')

                if data_type in ['probit_sm']:
                    for it in data_n.iter.unique():
                        ind = (data_n.iter == it) & (data_n.train_test == 'test')
                        max_it = data_n.loc[ind, 'r2'].max()
                        data_n.loc[data_n.iter == it, 'r2'] = (
                            max_it - data_n.loc[data_n.iter == it, 'r2'])

                is_legend = (i == 2) & (j == 0)
                plot_one(row_axes[j], data_n, is_legend)

        plt.savefig('../figures/{}.pdf'.format(data_type), bbox_inches='tight',
                    edgecolor='none', facecolor='none', dpi=100)

        plt.close()
