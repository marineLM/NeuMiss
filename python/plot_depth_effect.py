import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


def plot_one(ax, data):

    ax.axhline(0, color='.8', lw=2)

    # Plot train
    sns.lineplot(
        data=data.query('train_test == "train"'), x='n_params',
        y='r2', hue='method', ax=ax, ci=None, estimator=np.median,
        legend=False, style="train_test", dashes=6 * [(1, 2)])

    # Plot test
    sns.lineplot(
        data=data.query('train_test == "test"'), x='n_params',
        y='r2', hue='method', ax=ax, estimator=np.median,
        legend=False)

    # Set axes
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('{} score - Bayes rate'.format('R2'))
    #ax.grid(True)
    #sns.despine()

    ax.set_ylim(-.115, 0)
    ax.set_xscale('log')
    plt.yticks([0, -.05, -.1])
    ax.set_xlim(xmin=scores['n_params'].min(), xmax=4e4)

    # Plot a text giving the method
    plt.text(2e3, -.093, 'MLP Deep',
                color='C0',
                ha='left',
                va='top')
    plt.text(3.8e4, -.0525, 'MLP Wide',
                color='C1',
                ha='right',
                va='bottom')
    plt.text(1600, -.009, 'Neumann',
                color='C2',
                ha='right',
                va='top')

    # Second legend for train vs test
    line1 = mlines.Line2D([], [], color='k', linestyle='-', label='Test set')
    line2 = mlines.Line2D([], [], color='k', linestyle=':', label='Train set')
    l = plt.legend(handles=[line1, line2], loc=(.57, .68),
                   handlelength=1, handletextpad=.1)
    ax.add_artist(l)


def n_params_shallow_mlp(n_features, n_hidden_units):
    n_layer1 = 2*n_features*n_hidden_units + n_hidden_units
    return n_layer1 + n_hidden_units + 1


def n_params_deep_mlp(n_features, n_hidden_layers):
    n_layer1 = 2*n_features**2 + n_features
    n_params = n_layer1
    for _ in range(1, int(n_hidden_layers)):
        n_params += n_features**2 + n_features
    n_params += n_features + 1
    return n_params


def n_params_neumann(n_features, depth):
    # Take into account beta, mu and intercept
    n_params = 2*n_features + 1
    # Take into account Sigma_mis_obs
    n_params += n_features**2
    # Take into account the Neumann iterations
    for _ in range(int(depth)):
        n_params += n_features**2
    return n_params


def n_params_func(df):
    if df.method == 'Neumann':
        return n_params_neumann(df.n_features, df.depth_or_width)
    elif df.method == 'MLP_deep':
        return n_params_deep_mlp(df.n_features, df.depth_or_width)
    elif df.method == 'MLP_shallow':
        return n_params_shallow_mlp(
            df.n_features, df.depth_or_width*df.n_features)


if __name__ == '__main__':

    scores = pd.read_csv('../results/mixture1_depth_effect.csv', index_col=0)

    # Separate Bayes rate from other methods performances
    br = scores.query('method == "BayesRate"')
    scores = scores.query('method != "BayesRate"')

    # Differentiate Neumann and Neumann flex and Neumann res
    flex = scores.flex.fillna(False)
    res = scores.residual_connection.fillna(False)
    neu = (scores.method == 'Neumann')

    ind_flex_res = (neu) & (flex) & (res)
    ind_flex = (neu) & (flex) & (~res)
    ind_res = (neu) & (~flex) & (res)

    scores.loc[ind_flex, 'method'] = 'Neumann_flex'
    scores.loc[ind_res, 'method'] = 'Neumann_res'
    scores.loc[ind_flex_res, 'method'] = 'Neumann_flex_res'

    # Differentiate shallow and deep MLP
    scores.loc[scores.duplicated(), 'method'] = 'MLP_shallow'
    scores.loc[scores.width > 1, 'method'] = 'MLP_shallow'
    scores.loc[scores.method == 'torchMLP', 'method'] = 'MLP_deep'

    # Choose the methods to be plotted
    methods = ['MLP_shallow', 'MLP_deep', 'Neumann',
                #'Neumann_flex', 'Neumann_res', 'Neumann_flex_res'
               ]
    scores = scores.query('method in @methods')

    # Create a column depth_or_width that will give the depth of Neumann
    # or width of the shallow MLP
    scores['depth_or_width'] = scores.depth.copy()
    scores.loc[scores.method == 'MLP_shallow', 'depth_or_width'] = (
        scores.width.loc[scores.method == 'MLP_shallow'])

    scores['n_params'] = [n_params_func(r) for r in scores.itertuples()]

    # Normalize by the Bayes rate
    for it in scores.iter.unique():
        br_it = br.loc[br.iter == it, 'r2']
        br_it = float(br_it)
        scores.loc[scores.iter == it, 'r2'] = (
            scores.loc[scores.iter == it, 'r2'] - br_it)

    fig, ax = plt.subplots(1, 1, figsize=(4.3, 2))
    plot_one(ax, scores)


    scores_deep_test = scores.query('(train_test == "test") '
            '& ((method == "MLP_deep") | (method == "Neumann"))')
    scores_deep_test = scores_deep_test.groupby(['method', 'depth']).median()
    scores_deep_test = scores_deep_test.reset_index()
    plt.scatter(scores_deep_test['n_params'],
            scores_deep_test['r2'],
            #c=scores_test['experiment'].map(colors),
            color='.6',
            s=3*(scores_deep_test['depth'] + 1),
            ec='k', linewidth=.5)

    scores_width_test = scores.query('(train_test == "test") '
            '& (method == "MLP_shallow")')
    scores_width_test = scores_width_test.groupby(['method', 'width']).median()
    scores_width_test = scores_width_test.reset_index()
    plt.scatter(scores_width_test['n_params'],
            scores_width_test['r2'],
            #c=scores_test['experiment'].map(colors),
            color='C1',
            s=3*(scores_width_test['width']),
            ec='k', linewidth=.5)

    # Add a legend for the disks
    legend_points = [
                    plt.scatter([], [],
                            s=3*(i + 1), marker='o',linewidth=.5,
                            edgecolor='k', facecolors='.6',
                            label='%i' % int(i))
                    for i in range(1, int(scores_deep_test['depth'].max() + 1),
                                    2)]
    l = plt.legend(handles=legend_points, scatterpoints=1,
                   title='Network\ndepth',
                   handlelength=1, handletextpad=.1,
                   borderpad=.3, fontsize=9, loc=(1.0, .0), frameon=False)
    ax.add_artist(l)

    # Add a legend for the disks
    legend_points = [
                    plt.scatter([], [],
                            s=3*(i + 1), marker='o',linewidth=.5,
                            edgecolor='k', facecolors='C1',
                            label='%i$\,d$' % int(i))
                    for i in [1, 3, 10, 30, 50]]
    plt.legend(handles=legend_points, scatterpoints=1,
                title='width',
                handlelength=1, handletextpad=.2,
                borderpad=.3, fontsize=9, loc=(1.23, .0), frameon=False)



    plt.subplots_adjust(left=.17, bottom=.22, right=.75, top=.97)
    plt.savefig('../figures/depth_effect.pdf',
                edgecolor='none', facecolor='none', dpi=100)

    plt.close()
