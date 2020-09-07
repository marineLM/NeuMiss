import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_one(ax, data):

    ax.axhline(0, color='.8', lw=2)

    palette = {'Neumann': 'C2', 'Neumann_res': 'C8', 'Neumann_analytic': 'C4'}

    # Plot train
    sns.lineplot(
        data=data.query('train_test == "train"'), x='depth',
        y='r2', hue='method', ax=ax, ci=None, palette=palette,
        legend=False, style="train_test", dashes=6 * [(1, 2)])

    # Plot test
    sns.lineplot(
        data=data.query('train_test == "test"'), x='depth', palette=palette,
        y='r2', hue='method', ax=ax, estimator=np.median)

    l = plt.legend()
    l = plt.legend(l.legendHandles[1:],
                   [h.get_label() for h in l.legendHandles[1:]], fontsize=9,
                   handlelength=1, borderaxespad=.2)
    # Add the legend, so that we can add another
    ax.add_artist(l)

    # Set axes
    ax.set_xlabel('depth')
    ax.set_xscale('log')
    ax.set_xlim(1, data['depth'].max())

    ax.set_ylabel('{} score - Bayes rate'.format('R2'))
    ax.set_ylim(-0.18, 0)

    ax.grid(True)


if __name__ == '__main__':

    scores = pd.read_csv('../results/simu_architecture.csv', index_col=0)

    # Separate Bayes rate from other methods performances
    br = scores.query('method == "Bayes_rate"')
    scores = scores.query('method != "Bayes_rate"')

    methods = ['Neumann', 'Neumann_res', 'Neumann_analytic']
    scores = scores.query('method in @methods')

    scores.train_test.fillna('test', inplace=True)

    # Adjust for the Bayes rate
    for it in scores.iter.unique():
        br_it = br.loc[br.iter == it, 'r2']
        br_it = float(br_it)
        scores.loc[scores.iter == it, 'r2'] = (
            scores.loc[scores.iter == it, 'r2'] - br_it)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    plot_one(ax, scores)

    plt.savefig('../figures/simu_architecture.pdf', bbox_inches='tight',
                edgecolor='none', facecolor='none', dpi=100)

    plt.close()
