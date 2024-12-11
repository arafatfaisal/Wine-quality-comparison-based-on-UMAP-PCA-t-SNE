import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
#Load the dataset
wine_data = pd.read_csv('E:\DSS\homework 5\winequality-red.csv')
# Separate features and target
features = wine_data.drop(columns=['quality'])
target = wine_data['quality']

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize dimensionality reduction methods
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42)
umap_reducer = umap.UMAP(n_components=2, random_state=42)

# Apply each method
pca_result = pca.fit_transform(features_scaled)
tsne_result = tsne.fit_transform(features_scaled)
umap_result = umap_reducer.fit_transform(features_scaled)

# Prepare results for plotting
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])

# Add target (quality) as a label for color-coding
pca_df['quality'] = target
tsne_df['quality'] = target
umap_df['quality'] = target

# Plotting function
def plot_dimensionality_reduction(data, x, y, title, ax):
    sns.scatterplot(
        data=data, x=x, y=y, hue='quality', palette='viridis', ax=ax, s=40, alpha=0.8
    )
    ax.set_title(title)
    ax.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Plot each method's results
plot_dimensionality_reduction(pca_df, 'PC1', 'PC2', 'PCA Results', axes[0])
plot_dimensionality_reduction(tsne_df, 'Dim1', 'Dim2', 't-SNE Results', axes[1])
plot_dimensionality_reduction(umap_df, 'UMAP1', 'UMAP2', 'UMAP Results', axes[2])

plt.show()
