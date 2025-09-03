# Chama as biblitoecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
# Importação dos dados pelo csv
df = pd.read_csv("medical_examination.csv")

# 2
# Coluna "overweight" usa BMI (IMC) peso / altura elevado a 2.
height_m = df["height"] / 100
bmi = df["weight"] / (height_m ** 2)
df["overweight"] = (bmi > 25).astype(int)

# 3
# Normalização glicose e colesterol (0 = bom, 1 = ruim)
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


# 4
def draw_cat_plot():
    # 5
    # Chama as categorias
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # 6
    # Agrupar e contar ocorrências
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7
    # Plot do grafico por categória (colunas separadas por cardio)
    g = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar"
    )

    # 8
    # Pegar a figura do catplot grafico
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # Limpeza dos dados
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ].copy()

    # 12
    # Matriz de correlação
    corr = df_heat.corr(numeric_only=True)

    # 13
    # Máscara do triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    # Figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    # Heatmap (Mapa de calor) mais proximo de 1 = uma correlação positiva, mais proximo de -1 = uma correlação negativa
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
