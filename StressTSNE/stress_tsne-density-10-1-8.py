import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

def compute_tsne(vectors, perplexity=30, n_iter=1000, learning_rate=200):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                max_iter=n_iter, learning_rate=learning_rate)
    return tsne.fit_transform(vectors)

def plot_density_scatter(embedding, data, color_map, cmap_map, title=None):
    risk_descriptions = {
        10: "Курсанти",
        1: "Мобілізовані",
        8: "Фронтовики"
    }

    plt.figure(figsize=(14, 12))
    plt.rcParams["font.family"] = "Times New Roman"

    # Сірі точки — загальний фон
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c="#E8E8E8", s=10, alpha=0.4, edgecolors='none')

    for risk_value, color in color_map.items():
        indices = data.astype(int) == risk_value
        x = embedding[indices, 0]
        y = embedding[indices, 1]

        if len(x) == 0:
            continue

        # Щільність
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # Відображення з градацією
        sc = plt.scatter(x, y, c=z,
                         cmap=cmap_map.get(risk_value, "Greys"),
                         s=30, edgecolors="k", linewidths=0.2)

        # Кольорова шкала
        cbar = plt.colorbar(sc)
        cbar.set_label(f"Щільність: {risk_descriptions.get(risk_value, 'norm')}", fontsize=15)

    # Легенда
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[10],
               markerfacecolor=color_map[10], markersize=10, markeredgecolor="k"),
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[1],
               markerfacecolor=color_map[1], markersize=10, markeredgecolor="k"),
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[8],
               markerfacecolor=color_map[8], markersize=10, markeredgecolor="k"),
        Line2D([0], [0], marker='o', color='w', label='інші',
               markerfacecolor="#E8E8E8", markersize=10, markeredgecolor="none"),
    ]
    plt.legend(handles=legend_elements, fontsize=15, loc="best", frameon=True, fancybox=True)

    # Оформлення
    plt.xlabel("t-SNE-1", fontsize=15)
    plt.ylabel("t-SNE-2", fontsize=15)
    if title:
        plt.title(title, fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Load Excel file
file_path = "combined_dataset-selected-0.xlsx"
df = pd.read_excel(file_path)

# Drop completely empty columns and rows
dfx = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

# Convert to numeric if necessary
df = df.apply(pd.to_numeric, errors='coerce').fillna("")

# Fill missing values with the column median
df.fillna(df.median(numeric_only=True), inplace=True)

# Separate features and target
dfx = df.drop(columns=["STRESS"])
dfy = df["STRESS"]

# t-SNE
embedding = compute_tsne(dfx)

# Visualization
color_map = {
    10: "#1790D6",    
    1: "#00FF44",
    8: "#FFA500"  
}
cmap_map = {
    10: "Blues",
    1: "Greens",
    8: "Oranges"
}

plot_density_scatter(embedding, dfy, color_map, cmap_map,
                        title="Розподіл військових за комплексом параметрів ЕКГ/ВСР")
