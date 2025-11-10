from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

args = [
    "evaluate",
    "d_model",
    "dim_feedforward",
    "num_encoder_layers",
    "num_decoder_layers",
    "length_in",
    "length_tgt",
    "batch_size",
]

df = pd.read_csv("benchmark_transformer.csv")
df = df[df.length_tgt != 2]
df = df.drop("Unnamed: 0", axis=1)
val_args = [c for c in df.columns if c not in args]
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
print(df.groupby(args).mean())


def is_outlier(s):
    lower_limit = s.mean() - (s.std() * 2)
    upper_limit = s.mean() + (s.std() * 2)
    return ~s.between(lower_limit, upper_limit)


df = df[~df.groupby(args, group_keys=False)["time_encoder"].apply(is_outlier)]

df = df.melt(id_vars=args, value_vars=val_args)
# g = sns.relplot(kind='line', data=df, x='batch_size', y='value',
#                 style='evaluate', hue='variable', col='length_in', row='length_tgt',lw=3)
# g = sns.relplot(kind='line', data=df, x='length_in', y='value',
#                 style='evaluate', hue='variable', col='batch_size', row='length_tgt', lw=3)
g = sns.relplot(
    kind="line",
    data=df,
    x="length_in",
    y="value",
    style="evaluate",
    hue="length_tgt",
    hue_norm=LogNorm(),
    row="variable",
    col="batch_size",
    lw=3,
)
g.set(xscale="log", yscale="log")
plt.savefig(Path("export") / "benchmark_transformer.pdf", bbox_inches="tight")
plt.show()
