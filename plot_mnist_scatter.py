import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import ListedColormap

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_full, y_full = mnist.data, mnist.target.astype(int)
X_scaled = X_full / 255.0

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- Choose subset of labels ---
selected_labels = [1, 2, 3]  # Adjust this list as needed
n_labels = len(selected_labels)

# --- Filter main MNIST data ---
mask_main = np.isin(y_full, selected_labels)
X_pca_filtered = X_pca[mask_main]
y_filtered = y_full[mask_main]

# --- Set up figure with adjusted layout for colorbar on the right ---
fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(right=0.85)  # Make space for the colorbar on the right

# Main axes for scatter plot
main_ax = fig.add_subplot(111)
main_ax.axis('off')

# Create a discrete colormap for categorical coloring
cmap = ListedColormap(plt.cm.tab10.colors[:n_labels])

scatter = main_ax.scatter(
    X_pca_filtered[:, 0], X_pca_filtered[:, 1],
    c=y_filtered,
    cmap=cmap,
    alpha=0.6,
    s=5,
    vmin=min(selected_labels)-0.5,
    vmax=max(selected_labels)+0.5
)

# --- Add vertical colorbar on the right ---
cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(
    scatter,
    cax=cax,
    orientation='vertical',
    ticks=selected_labels  # Only show selected labels
)
#cbar.set_label('Digit', labelpad=10)

# --- Load and filter additional data ---
X = np.load(r"mnist_data\mnist\mnist_samp_9.npy")

samp_size = 700
X = X[:samp_size, :]

mnist_coordinates = X[:, :-2]
labels = X[:, -2].astype(int)
weights = X[:, -1]

# Filter additional data to selected labels
mask_additional = np.isin(labels, selected_labels)
mnist_coordinates = mnist_coordinates[mask_additional]
labels = labels[mask_additional]
weights = weights[mask_additional]

# Transform additional coordinates using PCA
mnist_coords_scaled = mnist_coordinates / 255.0
additional_pca = pca.transform(mnist_coords_scaled)

# --- Overlay images with adjusted size ---
size_scale = 200  # Increase if images are too small

for i, (coord, weight) in enumerate(zip(additional_pca, weights)):
    img = mnist_coordinates[i].reshape(28, 28) / 255.0
    im = OffsetImage(img, cmap='gray', zoom=weight * size_scale)
    ab = AnnotationBbox(im, coord, frameon=False)
    main_ax.add_artist(ab)

# Optional: Zoom into the region with overlaid images
pad = 1
x_min, x_max = additional_pca[:, 0].min()-pad, additional_pca[:, 0].max()+pad
y_min, y_max = additional_pca[:, 1].min()-pad, additional_pca[:, 1].max()+pad
main_ax.set_xlim(x_min, x_max)
main_ax.set_ylim(y_min, y_max)

plt.show()