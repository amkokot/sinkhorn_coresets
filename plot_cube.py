import numpy as np
import matplotlib.pyplot as plt
import os

# Define the r values
r_list = [10, 25, 50, 100, 250, 500, 1000]

# Define the methods (excluding the one that was not run)
methods = ['CO2 - Recomb.', 'Random', 'Halton', 'CO2 - Herding']

# Initialize arrays to store the averaged results and variances
# Shape: (num_r_values, num_methods)
num_r_values = len(r_list)
num_methods = len(methods)
all_results = np.zeros((num_r_values, num_methods))  # For means
all_squared_results = np.zeros((num_r_values, num_methods))  # For variance calculation
valid_counts = np.zeros((num_r_values, num_methods))  # For counting valid (non-NaN) results

# Load and process results over job_id 1-1000
for job_id in range(1, 1001):
    file_path = f'cube/cube_{job_id}.npy'
    try:
        results = np.loadtxt(file_path, delimiter=',')
        if not np.isnan(results).any():
            # Update the running totals for mean and variance
            all_results += results[:, [0, 1, 2, 4]]  # Exclude the unused method (index 3)
            all_squared_results += results[:, [0, 1, 2, 4]]**2
            valid_counts += 1
    except FileNotFoundError:
        print(f"File not found for job_id {job_id}. Skipping...")

# Compute the mean and variance
all_results /= valid_counts  # Mean
all_variances = (all_squared_results / valid_counts) - (all_results**2)  # Variance
all_std_errors = np.sqrt(all_variances / valid_counts)  # Standard error of the mean

# Define a more sophisticated color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the averaged results for each method with error bars

colors = ['C0', 'C1', 'red', 'purple']

re_order = [0, 3, 1, 2]
for i, method in enumerate(methods):
    j = re_order[i]
    plt.plot(r_list, all_results[:, j], label=methods[j], color=colors[i], marker='o')

# Add labels and title
plt.xlabel('Coreset Size', fontsize=16)
plt.ylabel('$S_\\varepsilon$ error', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
#plt.title('Comparison of Sinkhorn Loss Across Methods', fontsize=14)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.6)

# Add legend
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)

plt.legend(fontsize=12)

# Use a logarithmic scale for the x-axis if the r values span several orders of magnitude
plt.xscale('log')
plt.yscale('log')

plt.tick_params(axis='both', which='major', labelsize=10)

# Adjust layout to make room for the legend
plt.tight_layout()

# Show the plot
plt.show()