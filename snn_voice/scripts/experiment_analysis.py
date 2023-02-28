import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import tensorboard as tb


def load_experiment_metrics(experiment_id: str = "6JVd3wW1QFKmXTQzBPTOcw"):
    # Load in the cloud tensorboard experiment
    # See: https://tensorboard.dev/experiment/6JVd3wW1QFKmXTQzBPTOcw
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)

    # Get the metrics
    df: pd.DataFrame = experiment.get_scalars()

    # Remove unused tags
    df = df[(df['tag'] != 'epoch') & (df['tag'] != 'hp_metric')].reset_index(drop=True)

    # Split the 'run' string into a meta dataframe to append to the current df
    # Format: ___\\<Exp. Time>\\<Model Name>\\___.learn_beta=<LEARN BETA>,___.n_steps=<N STEPS>\\___\\___'
    # ___ can be ignored
    meta_list = []
    for i, r in df.iterrows():
        meta = r['run'].split("\\")
        datetime = meta[1]
        model = meta[2]

        if len(meta) == 6:  # This means it's an SNN model
            learn_beta, n_steps = [arg.split('=')[-1] for arg in meta[3].split(",")]
            learn_beta = learn_beta != 'False'
            n_steps = int(n_steps)
        else:
            learn_beta = None
            n_steps = None

        meta_list.append({
            'datetime': datetime,
            'model': model,
            'learn_beta': learn_beta,
            'n_steps': n_steps,
        })
    df_meta = pd.DataFrame(meta_list)
    df_meta['learn_beta'] = df_meta['learn_beta'].astype(bool)
    df = pd.concat([df_meta, df.drop('run', axis=1)], axis=1)
    df = df.sort_index()

    # # Get the maximum values of each tag
    # df_max = df.groupby(['model', 'learn_beta', 'n_steps', 'tag'], dropna=False).max().reset_index()
    # df_metrics = df_max.pivot(index=['model', 'learn_beta', 'n_steps'], columns='tag', values='value')
    # df_metrics_acc: pd.DataFrame = df_metrics[
    #     ['top1_Acc', 'top2_Acc', 'top5_Acc', 'val_top1_acc', 'val_top2_acc', 'val_top5_acc']
    # ]

    return df


df = load_experiment_metrics()
df = df.set_index(['model', 'tag']).sort_index()

sns.color_palette('rainbow')
fig, axs = plt.subplots(3, 4, sharex='all', sharey='all', figsize=[16, 10])

metrics = {'val_top1_acc': 'Top-1 Acc',
           'val_top2_acc': 'Top-2 Acc',
           'val_top5_acc': 'Top-5 Acc'}
for ax_y, (metric, metric_name) in zip(axs, metrics.items()):
    for model_snn, model_cnn, ax in zip(
            ['m5_snn', 'piczak_snn', 'hjh_snn', 'tcy_snn'],
            ['m5_cnn', 'piczak_cnn', 'hjh_cnn', 'tcy_nn'],
            ax_y):
        df_snn = df.loc[model_snn, metric]
        sns.lineplot(df_snn, x='step', y='value', hue='n_steps', errorbar=None, marker='o', ax=ax, palette='husl')
        df_cnn = df.loc[model_cnn, metric]
        sns.lineplot(df_cnn, x='step', y='value', color='black', errorbar=None, marker='o', ax=ax)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=False))
        ax.set_title(f"{model_snn.split('_')[0]} - {metric_name}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Accuracy")

plt.tight_layout()
plt.show()
