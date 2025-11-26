import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# enter here the input file
file_path = "combined_dataset-selected-ukr-db.xlsx"

def compute_umap(vectors):
    um = umap.UMAP(n_components=2, random_state=42)
    return um.fit_transform(vectors)

def plot_density_scatter(embedding, data, color_map, cmap_map, title=None):
    risk_descriptions = {
        2: "UA Norm",
        1: "Сadets",
        7: "Wounded",
        8: "Frontline",
        9: "PTSD signs"
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

        if risk_value == 9:
            plt.scatter(x, y,
                c=color, s=10, alpha=0.4, edgecolors='black')
        else:
            # Щільність
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)

            # Відображення з градацією
            sc = plt.scatter(x, y, c=z,
                            cmap=cmap_map.get(risk_value, "Greys"),
                            s=30, edgecolors="k", linewidths=0.2)

            # Кольорова шкала
            cbar = plt.colorbar(sc)
            cbar.set_label(f"Density: {risk_descriptions.get(risk_value, 'norm')}", fontsize=15)

    # Легенда
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[2],
               markerfacecolor=color_map[2], markersize=8, markeredgecolor="k"),
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[1],
               markerfacecolor=color_map[1], markersize=8, markeredgecolor="k"),
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[7],
               markerfacecolor=color_map[7], markersize=8, markeredgecolor="k"),
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[8],
               markerfacecolor=color_map[8], markersize=8, markeredgecolor="k"),
        Line2D([0], [0], marker='o', color='w', label=risk_descriptions[9],
               markerfacecolor=color_map[9], markersize=4, markeredgecolor="black"),
        Line2D([0], [0], marker='o', color='w', label='other',
               markerfacecolor="#E8E8E8", markersize=4, markeredgecolor="none"),
    ]
    plt.legend(handles=legend_elements, fontsize=15, loc="lower right", frameon=True, fancybox=True)

    # Оформлення
    plt.xlabel("UMAP-1", fontsize=15)
    plt.ylabel("UMAP-2", fontsize=15)
    if title:
        plt.title(title, fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Load Excel file
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

# UMAP
embedding = compute_umap(dfx)

# Visualization
color_map = {
    2: "#B9B9B9",
    1: "#4daf4a",
    7: "#1790D6",    
    8: "#FFA500",
    9: "#FF0000"  
}
cmap_map = {
    2: "Grays",
    1: "Greens",
    7: "Blues",
    8: "Oranges",
    9: "Reds"
}

plot_density_scatter(embedding, dfy, color_map, cmap_map,
                        title="Dataset Density by Types")
