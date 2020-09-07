import itertools

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np


def plot_one(ax, data, is_legend=False):

    if is_legend:
        is_legend = 'full'

    #sns.set_palette('dark')
    sns.set_palette(list(itertools.chain(*zip(
                            sns.husl_palette(5, l=.75),
                            sns.husl_palette(5, l=.5),
                            sns.husl_palette(5, l=.25),
                    ))))

    # Plot test
    sns.lineplot(
        data=data.query('train_test == "test"'), x='params_by_samples',
        y='r2', hue='experiment', ax=ax, estimator=np.median, legend=is_legend)

    if is_legend:
        l = plt.legend()
        l = plt.legend(l.legendHandles[1:],
                [h.get_label() for h in l.legendHandles[1:]], fontsize=9,
                    handlelength=1, borderaxespad=.2,
                    ncol=3, loc=(-.1, -.635), frameon=False)
        # Add the legend, so that we can add another
        ax.add_artist(l)

    # Plot train
    sns.lineplot(
        data=data.query('train_test == "train"'), x='params_by_samples',
        y='r2', hue='experiment', ax=ax, ci=None,
        legend=False, style="train_test", dashes=6 * [(1, 2)])

    # Set axes
    ax.set_xlabel('# params / # samples', labelpad=.2)
    ax.set_ylabel('{} score - Bayes rate'.format('R2'))
    ax.grid(True)

    #ax.set_xlim(0, 10)
    #ax.set_ylim(-0.28, 0)
    ax.set_xlim(data['params_by_samples'].min(),
                data['params_by_samples'].max())
    ax.set_xscale('log')


def n_params_neumann(n_features, depth):
    # Take into account beta, mu and intercept
    n_params = 2*n_features + 1
    # Take into account Sigma_mis_obs
    n_params += n_features**2
    # Take into account the Neumann iterations
    for _ in range(int(depth)):
        n_params += n_features**2
    return n_params


if __name__ == '__main__':
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.major.pad'] = 1
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['xtick.major.size'] = 2

    for data_type in ['mixture1', 'gaussian_sm', 'probit_sm']:

        scores = pd.read_csv('../results/linear/' + data_type + '.csv',
                             index_col=0)
        scores_old = pd.read_csv('../results/simu_nets3/' + data_type + '.csv',
                             index_col=0)
        scores = pd.concat([scores, scores_old.query('n == 10000')])
        scores = scores.sort_values(by='n')

        # Separate Bayes rate from other methods performances
        if data_type not in ['gaussian_m_es']:
            br = scores.query('method == "BayesRate"')
            scores = scores.query('method != "BayesRate"')

        # Differentiate Neumann and Neumann flex
        flex = scores.flex.fillna(False)
        res = scores.residual_connection.fillna(False)
        pretrain = scores.layerwise_pretraining.fillna(False)
        neu = (scores.method == 'Neumann')

        ind_flex_res = (neu) & (flex) & (res) & (~pretrain)
        ind_flex_res_pretrain = (neu) & (flex) & (res) & (pretrain)
        ind_flex = (neu) & (flex) & (~res) & (~pretrain)
        ind_flex_pretrain = (neu) & (flex) & (~res) & (pretrain)
        ind_res = (neu) & (~flex) & (res) & (~pretrain)
        ind_pretrain = (neu) & (~flex) & (~res) & (pretrain)
        ind_res_pretrain = (neu) & (~flex) & (res) & (pretrain)

        scores.loc[ind_flex, 'method'] = 'Neumann_flex'
        scores.loc[ind_res, 'method'] = 'Neumann_res'
        scores.loc[ind_flex_res, 'method'] = 'Neumann_flex_res'
        scores.loc[ind_flex_pretrain, 'method'] = 'Neumann_flex_pretrain'
        scores.loc[ind_flex_res_pretrain, 'method'] = 'Neumann_flex_res_pretrain'
        scores.loc[ind_pretrain, 'method'] = 'Neumann_pretrain'
        scores.loc[ind_res_pretrain, 'method'] = 'Neumann_res_pretrain'

        # Choose the methods to be plotted
        methods = ['Neumann']
        scores = scores.query('method in @methods')

        scores['params_by_samples'] = [
                n_params_neumann(r.n_features, r.depth) / r.n
                for r in scores.itertuples()]
        n_samples_factor = (np.round(scores.n / (10**(np.log10(scores.n)).astype(int)))
                            ).astype(int).apply('n=${}\cdot 10'.format)
        n_samples_exp = np.log10(scores.n).astype(int).apply('^{}$'.format)
        n_features_str = scores.n_features.astype(int).apply(', d={}'.format)
        scores['experiment'] = n_samples_factor.str.cat(n_samples_exp).str.cat(n_features_str)

        fig, axes = plt.subplots(1, 1, figsize=(4.6, 2.5))

        n_features = scores.n_features.unique()
        n_features = n_features[n_features < 60]
        n_samples = scores.n.dropna().unique()

        scores_with_br = list()
        for i, p in enumerate(n_features):
            data = scores.query('n_features == @p')

            # Compute the performances ajusted for the Bayes rate
            if data_type not in ['probit_sm']:
                for it in data.iter.unique():
                    br_it = br.loc[(br.iter == it) & (br.n_features == p), 'r2']
                    br_it = float(br_it)
                    data.loc[data.iter == it, 'r2'] = (
                        data.loc[data.iter == it, 'r2'] - br_it)

            else:
                for it in data.iter.unique():
                    ind = (data.iter == it) & (data.train_test == 'test')
                    max_it = data.loc[ind, 'r2'].max()
                    data.loc[data.iter == it, 'r2'] = (
                        data.loc[data.iter == it, 'r2'] - max_it)

            scores_with_br.append(data)

        scores_with_br = pd.concat(scores_with_br)
        plot_one(axes, scores_with_br,
                 is_legend=True)

        scores_test = scores_with_br.query('train_test == "test"')
        scores_test = scores_test.groupby(['experiment', 'depth']).median()
        scores_test = scores_test.reset_index()
        colors = dict(zip(scores_test['experiment'].unique(),
                          sns.color_palette()))
        plt.scatter(scores_test['params_by_samples'],
                scores_test['r2'],
                #c=scores_test['experiment'].map(colors),
                color='.6',
                s=3*(scores_test['depth'] + 1),
                ec='k', linewidth=.5)

        legend_points = [
                        plt.scatter([], [],
                              s=3*(i + 1), marker='o',linewidth=.5,
                              edgecolor='k', facecolors='.6',
                              label='%i' % int(i))
                        for i in range(0, int(scores_test['depth'].max() + 1),
                                       2)]
        plt.legend(handles=legend_points, scatterpoints=1,
                   title='Neumann\nnetwork\ndepth',
                   handlelength=1, handletextpad=.1,
                   borderpad=.3, fontsize=9, loc=(1.01, -.07), frameon=False)

        plt.yticks([0, -.1, -.2])
        plt.tight_layout(pad=.02, rect=[0, .23, 1, 1])
        plt.savefig('../figures/scaling_{}.pdf'.format(data_type),
                    edgecolor='none', facecolor='none', dpi=100)


        plt.close()
