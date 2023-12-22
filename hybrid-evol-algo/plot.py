import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def add_cost_values(ax):
    for p in ax.patches:
        y_offset = 250
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x(), p.get_height() + y_offset), fontsize=8)

algorithms = ["MSLS", "ILS", "LSNS_LS", "LSNS_noLS", "HEA_oper1_LS", "HEA_oper1_noLS", "HEA_oper2_LS", "HEA_oper2_noLS"]
instances = ["TSPA", "TSPB", "TSPC", "TSPD"]

costs_df = pd.read_csv('./hybrid-evol-algo/costs.csv')
costs_df.drop(costs_df.columns[0], axis=1, inplace=True)


df_min = costs_df[costs_df.Algorithm.isin(algorithms)].copy()
df_min[instances] = df_min[instances].applymap(lambda x: float(re.search(r"\((\d+\.\d+)", x).group(1)) 
                                  if re.search(r"\((\d+\.\d+)", x) else None)
df_min.index = algorithms

df_avg = costs_df[costs_df.Algorithm.isin(algorithms)].copy()
df_avg[instances] = df_avg[instances].applymap(lambda x: float(re.search(r"^\d+\.\d+", x).group()) 
                                  if re.search(r"^\d+\.\d+", x) else None)
df_avg.index = algorithms


viridis_colors = [cm.viridis(i/8) for i in range(8)]
scaling = [60000, 60000, 30000, 30000]

fig, axes = plt.subplots(2, 4, figsize=(25, 15))
bar_width = 0.8

for ax, column, ymin in zip(axes[0], instances, scaling):
    df_min.plot(kind='bar', y=column, ax=ax, title=f'Min Cost - {column}', legend=False, 
                color=viridis_colors, width=bar_width)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_ylabel("Minimum Cost")
    ax.set_ylim(ymin)
    ax.grid(True, linestyle='--')
    add_cost_values(ax)

for ax, column, ymin in zip(axes[1], instances, scaling):
    df_avg.plot(kind='bar', y=column, ax=ax, title=f'Avg Cost - {column}', legend=False, 
                color=viridis_colors, width=bar_width)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_ylabel("Average Cost")
    ax.set_ylim(ymin)
    ax.grid(True, linestyle='--')
    add_cost_values(ax)

fig.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.3)

plt.savefig("./hybrid-evol-algo/plots/costs_bar_plot.png")
