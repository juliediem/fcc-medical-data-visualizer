import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Functions
# Check if overweight
def overweight_check(weight, height):
    height = height / 100
    bmi = (weight / height ** 2)
    if bmi > 25:
        return 1
    else:
        return 0


# Normalize data to 0 or 1 values
def normalize_column(col):
    if col == 1:
        return 0
    elif col > 1:
        return 1


# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df.apply(lambda row: overweight_check(row['weight'], row['height']), axis=1)

# 3
df['cholesterol'] = df['cholesterol'].apply(normalize_column)
df['gluc'] = df['gluc'].apply(normalize_column)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['id', 'cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.sort_values(by='variable')

    # 7
    g = sns.catplot(x='variable', hue='value', kind='count', col='cardio', data=df_cat)
    g.set_ylabels('total')

    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df["height"] >= df["height"].quantile(0.025)) &
                 (df["height"] <= df["height"].quantile(0.975)) &
                 (df["weight"] >= df["weight"].quantile(0.025)) &
                 (df["weight"] <= df["weight"].quantile(0.975))]

    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(df_heat.corr(), dtype=np.bool_))

    # 14
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='rocket', ax=ax)
    plt.xticks(rotation=90)
    ax.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)

    # 15

    # 16
    fig.savefig('heatmap.png')
    return fig
