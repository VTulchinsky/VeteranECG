import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_violin_from_xlsx(file_path, attributes, class_1, class_2, 
                          class_1_name, class_2_name, color_1, color_2):
    # Load the dataset
    df = pd.read_excel(file_path)

    # Assume the first column contains class labels
    class_column = df.columns[0]
    attribute_columns = attributes

    # Filter and rename class labels
    df_filtered = df[df[class_column].isin([class_1, class_2])].copy()
    class_name_map = {class_1: class_1_name, class_2: class_2_name}
    df_filtered[class_column] = df_filtered[class_column].map(class_name_map)

    # Melt the dataframe for plotting
    df_melted = df_filtered[[class_column] + attribute_columns].melt(id_vars=class_column,
                                                                      value_vars=attribute_columns,
                                                                      var_name='Attribute',
                                                                      value_name='Value')

    # Create violin plot with custom colors
    plt.figure(figsize=(10, 6))
    palette = {class_1_name: color_1, class_2_name: color_2}
    sns.violinplot(x='Attribute', y='Value', hue=class_column, data=df_melted,
                   split=True, palette=palette)
    plt.title(f'Parameter Distribution: {class_1_name} vs {class_2_name}')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.show()

# Example usage
xlsx_file = 'combined_dataset-selected-ukr-db.xlsx'
selected_attributes = ['HEART_RATE', 'SDNN', 'STRESS_INDEX', 'TRIANGULAR_INDEX', 'STAMINA']
class_A = 9
class_B = 8
class_A_name = 'PTSD Signs'
class_B_name = 'Frontline'
color_A = '#FF0000'
color_B = "#FFA500"

plot_violin_from_xlsx(xlsx_file, selected_attributes, 
                      class_A, class_B, class_A_name, class_B_name, 
                      color_A, color_B)