import numpy as np
from matplotlib import pyplot as plt

n_list = [5000, 10000, 15000, 25000]
d_list = [2, 5, 10]
r_list = [10, 25, 50, 100, 250, 500, 1000]
max_ind = 560

wine = np.array([136, 34, 85]) / 256

intensity = 0.27

# Load data
data = np.zeros((max_ind, 4, 7))

for i in range(max_ind):
    file_name = f"gauss/gauss_1_gpu_{i+1}.npy"
    with open(file_name, 'r') as f:
        lines = f.readlines()
    data_array = [list(map(float, line.strip().split(','))) for line in lines]
    data[i, :, :] = np.array(data_array).T

# Process data for varying d
change_sink_d = np.zeros((len(d_list), len(r_list)))
change_rand_d = np.zeros((len(d_list), len(r_list)))
tot_d = np.zeros(len(d_list))

for i in range(1, max_ind):
    if (i + 1) % 4 == 3:
        j = (i + 1) % 3
        tot_d[j] += 1
        change_sink_d[j] += data[i, 2, :]
        change_rand_d[j] += data[i, 3, :]

change_sink_d /= tot_d[:, None]
change_rand_d /= tot_d[:, None]

# Process data for varying n
change_sink_n = np.zeros((len(n_list), len(r_list)))
change_rand_n = np.zeros((len(n_list), len(r_list)))
tot_n = np.zeros(len(n_list))

for i in range(1, max_ind):
    if (i + 1) % 3 == 2:
        j = (i + 1) % 4
        tot_n[j] += 1
        change_sink_n[j] += data[i, 2, :]
        change_rand_n[j] += data[i, 3, :]

change_sink_n /= tot_n[:, None]
change_rand_n /= tot_n[:, None]

# Compute ratios
ratios_d = change_sink_d / change_rand_d
ratios_n = change_sink_n / change_rand_n
global_y_min = min(ratios_d.min(), ratios_n.min())

# Create figure with shared y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
linestyles = ["-", "-", "-", "-"]
markers = ["o", "s", "D", "^"]

colors = ['C0', 'C1', 'red', 'purple']
x_fill = [1e-10, 1e10]  # Extended x range for fill

ax1.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Left subplot (varying d)
for j in range(len(d_list)):
    ax1.plot(r_list, ratios_d[j], linestyle=linestyles[j], marker=markers[j],
             label=f'$d={d_list[j]}$', color = colors[j])
ax1.axhline(1, color='k', linestyle='--', linewidth=1)
ax1.fill_between(x_fill, 0, 1, color='lightgreen', alpha=intensity)
#ax1.fill_between(x_fill, 1, 100, color='lightcoral', alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Coreset Size', fontsize=16)
ax1.set_ylabel('Ratio of $S_\\varepsilon$ between CO2 and Random Samples', fontsize=16)
ax1.set_xlim(min(r_list)/1.1, max(r_list)*1.1)
ax1.set_ylim(global_y_min/10, 50)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=12)

# Right subplot (varying n)
for j in range(len(n_list)):
    ax2.plot(r_list, ratios_n[j], linestyle=linestyles[j], marker=markers[j],
             label=f'$n={n_list[j]}$', color = colors[j])
ax2.axhline(1, color='k', linestyle='--', linewidth=1)
ax2.fill_between(x_fill, 0, 1, color='lightgreen', alpha=intensity)
#ax2.fill_between(x_fill, 1, 100, color='lightcoral', alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Coreset Size', fontsize=16)
ax2.set_xlim(min(r_list)/1.1, max(r_list)*1.1)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=12)

from matplotlib.transforms import Affine2D

arrow_props = dict(arrowstyle="->", color=wine, lw=2, mutation_scale=15)
transform_flip_y = Affine2D().scale(1, -1) + ax2.transAxes
transform_flip = Affine2D().scale(1, -1).translate(0, 2) + ax2.get_xaxis_transform()
ax2.annotate("Rand Better", xy=(max(r_list)*1.1, 0.5), xytext=(max(r_list)*1.1, 1.4),
             fontsize=12, color=wine, ha='left', va='bottom', rotation=270, transform=transform_flip)
ax2.annotate("", xy=(max(r_list)*1.1, 50), xytext=(max(r_list)*1.1, 0.99),
             arrowprops=arrow_props, fontsize=14, color=wine, ha='center')

arrow_props = dict(arrowstyle="->", color="green", lw=2, mutation_scale=15)
ax2.annotate("CO2 Better", xy=(max(r_list)*1.1, 0.35), xytext=(max(r_list)*1.1, 0.05),
             fontsize=12, color="green", ha='left', va='bottom', rotation=270)
ax2.annotate("", xy=(max(r_list)*1.1, 0.02), xytext=(max(r_list)*1.1, 1.01),
             arrowprops=arrow_props, fontsize=14, color="green", ha='center')

plt.tight_layout()
plt.show()