from matplotlib import pyplot as plt
import numpy as np

n_list = [5000, 10000, 15000, 25000]
d_list = [2, 5, 10]
r_list = [10, 25, 50, 100, 250, 500, 1000]

max_ind = 1120

data = np.zeros((max_ind, 4, 7))

for i in range(max_ind):
    file_name = "gauss/gauss_1_kern_" + str(i+1) + ".npy"
    with open(file_name, "r") as f:
        lines = f.readlines()
    data_array = [list(map(float, line.strip().split(","))) for line in lines]
    data[i, :, :] = np.array(data_array).T

change_sink = np.zeros((2, len(r_list)))
change_rand = np.zeros((2, len(r_list)))
tot = 0

for i in range(1, max_ind):
    if (i+1) % 4 == 1:
        if (i+1) % 3 == 2:
            tot += 1
            change_sink[0, :] = data[i, 0, :] * n_list[(i+1) % 4]
            change_sink[1, :] = data[i, 2, :]
            change_rand[0, :] = data[i, 1, :] * n_list[(i+1) % 4]
            change_rand[1, :] = data[i, 3, :]

change_sink /= tot
change_rand /= tot

# Compute normalized absolute deviation for sink and rand
norm_dev_sink = np.abs(change_sink[0, :] - change_sink[1, :]) / np.abs(change_sink[0, :])
norm_dev_rand = np.abs(change_rand[0, :] - change_rand[1, :]) / np.abs(change_rand[0, :])

# Plotting
plt.figure(figsize=(8, 6))

# Plot normalized deviations
plt.plot(r_list, 100 * norm_dev_sink, label="Compression", marker="o", color="C0")
#plt.plot(r_list, norm_dev_rand, label="Random - Normalized Deviation", marker="s", color="C1")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=10)
# Improve plot appearance
plt.xlabel("Coreset Size", fontsize=16)
plt.ylabel("$2^{\\text{nd}}$ Ord. Aprroximation Percent Error", fontsize=16)
#plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Show the plot
plt.show()
