import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['pdf.fonttype'] = 42


def plot_one(ax, data):

    ax.axhline(0, color='.8', lw=2)

    # palette = {'Neumann': 'C2', 'Neumann_res': 'C8', 'Neumann_analytic': 'C4'}
    palette = {'NeuMiss': 'C2', 'NeuMiss_res': 'C8', 'NeuMiss_analytic': 'C4'}

    # Plot train
    sns.lineplot(
        data=data.query('train_test == "train"'), x='depth',
        y='r2', hue='method', ax=ax, ci=None, palette=palette,
        legend=False, style="train_test", dashes=6 * [(1, 2)])

    # Plot test
    sns.lineplot(
        data=data.query('train_test == "test"'), x='depth', palette=palette,
        y='r2', hue='method', ax=ax, estimator=np.median)

    ll = plt.legend()
    ll = plt.legend(ll.legendHandles[1:],
                    [h.get_label() for h in ll.legendHandles[1:]], fontsize=9,
                    handlelength=1, borderaxespad=.2)
    # Add the legend, so that we can add another
    ax.add_artist(ll)

    # Set axes
    ax.set_xlabel('depth')
    ax.set_xscale('log')
    ax.set_xlim(1, data['depth'].max())

    ax.set_ylabel('{} score - Bayes rate'.format('R2'))
    ax.set_ylim(-0.18, 0)

    ax.grid(True)


if __name__ == '__main__':

    scores = pd.read_csv('../results/simu_archi.csv', index_col=0)

    # Separate Bayes rate from other methods performances
    br = scores.query('method == "Bayes_rate"')
    scores = scores.query('method != "Bayes_rate"')
    scores.train_test.fillna('test', inplace=True)

    # Adjust for the Bayes rate
    for it in scores.iter.unique():
        for split in scores.train_test.unique():
            mask_br = (br.iter == it) & (br.train_test == split)
            br_val = br.loc[mask_br, 'r2']
            br_val = float(br_val)
            mask_data = (scores.iter == it) & (scores.train_test == split)
            scores.loc[mask_data, 'r2'] = scores.loc[mask_data, 'r2'] - br_val

    scores['method'] = scores['method'].replace({
            'Neumann_res': 'NeuMiss_res',
            'Neumann_analytic': 'NeuMiss_analytic',
            'Neumann': 'NeuMiss'
            })

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    plot_one(ax, scores)

    plt.savefig('../figures/simu_architecture.pdf', bbox_inches='tight',
                edgecolor='none', facecolor='none', dpi=100)

    plt.close()
