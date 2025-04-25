import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

reps = 500

e_vec = np.zeros((reps, 6))

for i in range(1, reps):
    if True:
        file_name = "mnist_data/mnist/mnist_og_fulldata_reps_" + str(i+1) + ".npy"
        e_vec[i, :] = np.load(file_name)
        # r = r_list[i % 3]
        # reg = reg_list[i % 6]

wine = np.array([136, 34, 85]) / 256

e_vec = e_vec[e_vec[:, 0] != 0, :]

# Define consistent colors
recombination_color = "C0"  # Blue
random_color = "C1"         # Orange

# ----------------------
# First Q-Q plot (Linear Scale)
# ----------------------
plt.figure(figsize=(8, 6))

# For a Q-Q plot, sort each array to obtain the quantiles.
x_data = np.sort(e_vec[:, 0])
y_data = np.sort(e_vec[:, 1])

# Determine axis limits from the quantile data.
x_max = np.max(x_data)
y_max = np.max(y_data)
max_val = max(x_max, y_max) * 1.05
x_min = np.min(x_data)
y_min = np.min(y_data)
min_val = min(x_min, y_min) * 0.9

# Plot the reference line y = x
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

# Fill background colors with triangular regions
plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='lightgreen', alpha=0.27)

# Plot the quantile data as a scatter plot
plt.scatter(x_data, y_data, alpha=0.8, color=recombination_color, edgecolor="black", linewidth=0.5)

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# Increase tick label font size
plt.tick_params(axis='both', which='major', labelsize=10)

# Add arrows and text
arrow_props = dict(arrowstyle="->", color="green", lw=2, mutation_scale=15)
plt.annotate("CO2 Better", xy=(0.8 * max_val, 1 * max_val), xytext=(0.906 * max_val, 1.01 * max_val),
             fontsize=14, color="green", ha='center')

plt.annotate("", xy=(0.79 * max_val, 1 * max_val), xytext=(1.0028 * max_val, 1.* max_val),
             arrowprops=arrow_props, fontsize=12, color="green", ha='center')


arrow_props = dict(arrowstyle="->", color=wine, lw=2, mutation_scale=15)

plt.annotate("Rand Better", xy=(1* max_val, 0.75 * max_val), xytext=(1* max_val, 0.77 * max_val),
             fontsize=14, color=wine, ha='left', va='bottom', rotation = 270)

plt.annotate("", xy=( 0.9985 * max_val, 0.72 * max_val), xytext=(0.9985 * max_val, 1.0028 * max_val),
             arrowprops=arrow_props, fontsize=12, color=wine, ha='center')

plt.xlabel(r"CO2: $\ell_1$ error", fontsize = 16)
plt.ylabel(r"Random: $\ell_1$ error", fontsize = 16)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tick_params(axis='both', which='major', labelsize=13)

# ----------------------
# Second Q-Q plot (Log Scale)
# ----------------------
plt.figure(figsize=(8, 6))

# Sort the quantiles for the second pair of arrays.
x_data = np.sort(e_vec[:, 4])
y_data = np.sort(e_vec[:, 5])

x_max = np.max(x_data)
y_max = np.max(y_data)
max_val = max(x_max, y_max) * 1.05
x_min = np.min(x_data)
y_min = np.min(y_data)
min_val = min(x_min, y_min) * 0.9

plt.plot([min_val, max_val], [min_val, max_val], 'k--')
plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='lightgreen', alpha=0.27)
plt.scatter(x_data, y_data, alpha=0.8, color=recombination_color, edgecolor="black", linewidth=0.5)

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.tick_params(axis='both', which='major', labelsize=10)

# Add arrows and text
arrow_props = dict(arrowstyle="->", color="green", lw=2, mutation_scale=15)
plt.annotate("CO2 Better", xy=(0.48 * max_val, 1 * max_val), xytext=(0.495 * max_val, 1.1 * max_val),
             fontsize=14, color="green", ha='center')

plt.annotate("", xy=(0.2 * max_val, 1 * max_val), xytext=(1.0028 * max_val, 1.* max_val),
             arrowprops=arrow_props, fontsize=12, color="green", ha='center')


arrow_props = dict(arrowstyle="->", color=wine, lw=2, mutation_scale=15)

plt.annotate("Rand Better", xy=(1* max_val, 0.2 * max_val), xytext=(1* max_val, 0.15 * max_val),
             fontsize=14, color=wine, ha='left', va='bottom', rotation = 270)

plt.annotate("", xy=( 0.9985 * max_val, 0.1 * max_val), xytext=(0.9985 * max_val, 1.0028 * max_val),
             arrowprops=arrow_props, fontsize=12, color=wine, ha='center')

plt.xlabel(r"CO2: $S_{\varepsilon}$ error", fontsize = 16)
plt.ylabel(r"Random: $S_{\varepsilon}$ error", fontsize = 16)
plt.xscale('log')
plt.yscale('log')
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=13)

plt.show()
