import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['pdf.fonttype'] = 42


def plot_one(ax, data, is_legend=False):

    if is_legend:
        is_legend = 'full'

    sns.set_palette('bright')

    # Plot test
    sns.boxplot(
        data=data.query('train_test == "test"'), x='score',
        palette={'EM': '.9', 'MLP': 'C1', 'NeuMiss': 'C2',
                 'MICE + MLP': 'C3', 'MICE + LR': 'C3', },
        saturation=1,
        y='method', ax=ax)

    for i in range(len(data['method'].unique())):
        if i % 2:
            ax.axhspan(i - .5, i + .5, color='.9', zorder=0)

    # Set axes
    ax.set_ylabel(None)


if __name__ == '__main__':
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['xtick.major.pad'] = .7
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['xtick.major.size'] = 2

    # Uncomment the option you want to choose
    # option = 'shared_scale'
    option = 'adapted_scale'

    for data_type in ['MCAR', 'MAR_logistic', 'gaussian_sm', 'probit_sm']:

        scores = pd.read_csv(f'../results/' + data_type + '.csv', index_col=0)

        # Keep results for n=2e4 and n=1e5
        scores = scores.query('n!=10000')

        # Separate Bayes rate from other methods performances
        if data_type not in ['probit_sm']:
            # option 1
            # br = scores.query('method == "BayesRate"')
            # scores = scores.query('method != "BayesRate"')
            # option 2
            br = scores.query('method == "BayesPredictor"')
            scores = scores.query('method != "BayesPredictor"')

        # Differentiate Neumann and Neumann res es
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
        methods = ['EMLR', 'MICELR', 'torchMLP', 'Neumann']
        scores = scores.query('method in @methods')

        n_samples_str = scores.n.astype(int).apply('n={}'.format)
        n_features_str = scores.n_features.astype(int).apply(', d={}'.format)
        scores['experiment'] = n_samples_str.str.cat(n_features_str)

        n_features = scores.n_features.unique()
        n_samples = scores.n.dropna().unique()

        scores_with_br = list()
        for i, p in enumerate(n_features):
            data = scores.query('n_features == @p')

            # Compute the performances ajusted for the Bayes rate
            # option 1: the Bayes rate is computed once per iteration and per p
            # (whatever n)
            # if data_type not in ['probit_sm']:
            #     for it in data.iter.unique():
            #         br_it = br.loc[
            #             (br.iter == it) & (br.n_features == p), 'r2']
            #         br_it = float(br_it)
            #         data.loc[data.iter == it, 'r2'] = (
            #             np.minimum(data.loc[data.iter == it, 'r2'] - br_it, 0))

            # option 2: the Bayes Predictor is just another method , and is
            # computed for each iter, n and p.
            for it in data.iter.unique():
                for n in data.n.unique():
                    for split in data.train_test.unique():
                        if data_type not in ['probit_sm']:
                            mask_br = ((br.n_features == p) & (br.iter == it) &
                                       (br.n == n) & (br.train_test == split))
                            br_val = br.loc[mask_br, 'r2']
                            br_val = float(br_val)
                            mask_data = ((data.n_features == p) &
                                         (data.iter == it) & (data.n == n) &
                                         (data.train_test == split))
                            data.loc[mask_data, 'r2'] = (
                                data.loc[mask_data, 'r2'] - br_val)
                        else:
                            mask_data = ((data.n_features == p) &
                                         (data.iter == it) & (data.n == n) &
                                         (data.train_test == split))
                            # constant added so that the scores are not too
                            # close to 0, otherwise the plot in logscale is
                            # difficult to read.
                            best_r2 = data.loc[mask_data, 'r2'].max()
                            data.loc[mask_data, 'r2'] = np.minimum(
                                -1e-3, data.loc[mask_data, 'r2'] - best_r2)

            if p >= 50:
                # EM failed over dim 50
                data.loc[data.method == 'EMLR', 'r2'] = np.nan

            scores_with_br.append(data)

        scores_with_br = pd.concat(scores_with_br)
        scores_with_br['method'] = scores_with_br['method'].replace({
            'EMLR': 'EM',
            'MICELR': 'MICE + LR',
            'torchMLP': 'MLP',
            'Neumann': 'NeuMiss'
            })

        scores_with_br['score'] = scores_with_br['r2']

        # Create a column depth_or_width that will give the depth of Neumann
        # or width of the shallow MLP
        scores_with_br['depth_or_width'] = scores_with_br.width.copy()
        scores_with_br.depth_or_width.fillna(
            scores_with_br.depth, inplace=True)
        scores_with_br.depth_or_width.fillna(0, inplace=True)

        # Find the best depth on validation set
        scores_with_br_val = scores_with_br.query('train_test == "val"')
        scores_with_br_val = scores_with_br_val.sort_values(
            by=['method', 'experiment', 'iter', 'score'],
            ascending=False)
        scores_with_br_val = scores_with_br_val.groupby(
            ['method', 'experiment', 'iter']).head(1)
        scores_with_br_val.rename(
            columns={'depth_or_width': 'best_depth'}, inplace=True)
        scores_with_br_val = scores_with_br_val[
            ['method', 'experiment', 'iter', 'best_depth']]
        scores_no_depth = scores_with_br.merge(scores_with_br_val)
        scores_no_depth = scores_no_depth.query('depth_or_width == best_depth')

        fig, axes = plt.subplots(len(n_features), len(n_samples),
                                 figsize=(3.5, 3.1), sharey=True)

        for i, p in enumerate(n_features):
            for j, n in enumerate(n_samples):
                data = scores_no_depth.query(
                    '(n == @n) & (n_features == @p)')
                ax = axes[i, j]
                plot_one(ax, data, is_legend=False)
                if not ((i == 2) and (j == 0)):
                    ax.set_xlabel(None)
                else:
                    if data_type not in ['probit_sm']:
                        ax.set_xlabel('R2 - Bayes rate', labelpad=1)
                    else:
                        ax.set_xlabel('R2 - (best R2)', labelpad=1)

                if data_type not in ['MCAR', 'MAR_logistic']:
                    ax.set_xscale('symlog', linthreshx=.002)

                    if option == 'adapted_scale':
                        xmin, xmax = ax.get_xlim()
                        xmin = max(xmin, -1.05)
                        xmax = min(xmax, -0.00101)
                        ax.set_xlim(xmin, xmax)
                        if (xmin < -.2) and (xmax < -.05):
                            xticks = (-.2, -.1)
                        elif (xmin < -.1) and (xmax < -.02):
                            xticks = (-.1, -.05)
                        elif (xmin < -.1) and (xmax < -.01):
                            xticks = (-.1, -.02)
                        elif (xmin < -.1) and (xmax < -.001):
                            xticks = (-.1, -.01)
                        else:
                            xticks = (-.1, -.01, -.001)
                        ax.set_xticks(xticks)
                    elif option == 'shared_scale':
                        if data_type == 'gaussian_sm':
                            ax.set_xlim(-0.5, -0.0025)
                            xticks = (-.1, -.01)
                        elif data_type == 'probit_sm':
                            ax.set_xlim(-0.5, -0.00101)
                            xticks = (-.1, -.01)
                        ax.set_xticks(xticks)

                    def fmt_flt(x):
                        if x <= -.1:
                            return '%.1f' % x
                        elif x == 0:
                            return '0'
                        elif x <= -.01:
                            return '%.2f' % x
                        else:
                            return '%.3f' % x
                    ax.set_xticklabels([fmt_flt(t) for t in xticks])

                elif data_type in ['MAR_logistic']:
                    if option == 'adapted_scale':
                        xmin, xmax = ax.get_xlim()
                        if p == 10 and n == 100000:
                            xmin = max(xmin, -0.15)
                            xmax = max(xmax, .001)
                        elif p == 10 and n == 20000:
                            xmin = max(xmin, -0.12)
                            xmax = max(xmax, .001)
                        ax.set_xlim(xmin, xmax=min(xmax, .000001))
                    elif option == 'shared_scale':
                        ax.set_xlim(-0.15, xmax=.000001)
                        ax.set_xticks((0, -.05, -.1))

                elif data_type in ['MCAR']:
                    if option == 'adapted_scale':
                        xmin, xmax = ax.get_xlim()
                        if n == 100000 and p == 10:
                            xmin = max(xmin, -0.095)
                            ax.set_xlim(xmin, xmax=min(xmax, .000001))
                    elif option == 'shared_scale':
                        ax.set_xlim(-0.195, xmax=.000001)
                        ax.set_xticks((0, -.1,))

                if j == 0:
                    plt.text(-.31, .85, 'd=%i' % p, ha='right',
                             transform=ax.transAxes, color='w',
                             bbox=dict(facecolor='k', pad=1),
                             )
                if i == 0:
                    plt.text(.98, 1.02, 'n=%i' % n, ha='right',
                             transform=ax.transAxes, color='w',
                             bbox=dict(facecolor='k', pad=1),
                             )
                sns.despine()

        plt.subplots_adjust(left=.225, bottom=.1, right=.965, top=.95,
                            wspace=.1, hspace=.25)
        plt.savefig('../figures/boxplot_{}_{}.pdf'.format(data_type, option),
                    edgecolor='none', facecolor='none', dpi=100)

        plt.close()
