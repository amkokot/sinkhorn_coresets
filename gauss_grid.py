from recombination import *
import matplotlib.pyplot as plt
from rp_cholesky import *
import numpy as np

n = 10000
reg = 0.75
eps = 1
mu = np.array([[3, 3], [3, -3], [-3,3], [-3,-3], [0, 6], [6, 0], [-6, 0], [0, -6]])

# Define grid size
pics = 3  # 3x3 grid, adjust as needed
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
axs = axs.flatten()  # Flatten for easy iteration

# Generate data (X) and heatmap (Z) once
k = 8  # From k_list
view_space = 8  # From view_list[j=0]
mesh = 250

# Generate X data
X = np.zeros((n, 2))
draws = np.random.multinomial(n, np.ones(k)/k)
tot = 0
for i in range(k):
    X[tot:tot+draws[i], :] = np.random.multivariate_normal(mu[i], np.eye(2), draws[i])
    tot += draws[i]

# Generate Z (heatmap)
x = np.linspace(-view_space, view_space, mesh)
y = np.linspace(-view_space, view_space, mesh)
Z = np.zeros((mesh, mesh))

def f(x_val, y_val, eps, mu, k):
    return sum(np.exp(-np.linalg.norm([x_val, y_val] - mu[i])**2 / eps) for i in range(k))

for h in range(mesh):
    for i in range(mesh):
        Z[h, i] = f(x[h], y[i], eps, mu, k)

# Plot each subplot
for idx in range(8):
    samp = idx + 1 + 20 # Replace with your samp indices if needed
    ax = axs[idx]
    
    # Load nodes

    with open(f"gauss/gauss_16_nodes_{samp}.npy", "r") as f:
        nodes = np.array([list(map(float, line.strip().split(","))) for line in f])

    
    with open(f"gauss/gauss_16_weights_{samp}.npy", "r") as f:
        weights = np.array([list(map(float, line.strip().split(","))) for line in f]).flatten()

    # Plot
    ax.imshow(Z, extent=(-view_space, view_space, -view_space, view_space), 
              origin='lower', cmap='Greys', alpha=0.7)
    ax.scatter(nodes[:,0], nodes[:,1], s=weights*300, label="Sinkhorn Compression")
    #ax.set_title(f"samp={samp}")
    ax.axis(False)

plt.tight_layout()
plt.show()