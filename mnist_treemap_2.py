import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import squarify
from sklearn.datasets import fetch_openml
from collections import defaultdict



samp_num = 11

def load_mnist_subset(num_samples=1000, random_state=42):
    X = np.load(r"mnist_data\mnist\mnist_samp_"  + str(samp_num) + ".npy")
    X = X[X[:, -1].argsort()]
    mnist_coords = X[:, :-2] / 255
    y = X[:, -2].astype('int')


    
    return mnist_coords, y

def assign_weights(y, method='chisquared'):
    X = np.load(r"mnist_data\mnist\mnist_samp_" + str(samp_num) + ".npy")
    X = X[X[:, -1].argsort()]
    weights = X[:, -1]
    
    return weights

def create_two_row_treemap(X, y, weights, figsize=(16, 10), max_images_per_digit=50, 
                          image_border_width=0.2, category_border_width=0.6):
    """Create a treemap with two fixed-height rows (0-4 and 5-9)."""
    # Group data by digit
    digit_groups = defaultdict(list)
    for i, digit in enumerate(y):
        digit_groups[digit].append((i, weights[i]))
    
    # Calculate total weight for each digit
    row1_digits = [0, 1, 2, 3, 4]  # First row
    row2_digits = [5, 6, 7, 8, 9]  # Second row
    
    row1_weights = {}
    row2_weights = {}
    
    # Get weights for each digit
    for digit in row1_digits:
        if digit in digit_groups:
            items = digit_groups[digit]
            row1_weights[digit] = sum(weight for _, weight in items)
        else:
            row1_weights[digit] = 0
            
    for digit in row2_digits:
        if digit in digit_groups:
            items = digit_groups[digit]
            row2_weights[digit] = sum(weight for _, weight in items)
        else:
            row2_weights[digit] = 0
    
    # Print the total weights per digit for verification
    print("Total weights per digit:")
    for digit in range(10):
        if digit in row1_weights:
            print(f"Digit {digit}: {row1_weights[digit]:.2f}")
        elif digit in row2_weights:
            print(f"Digit {digit}: {row2_weights[digit]:.2f}")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get color map (using the recommended approach for matplotlib 3.7+)
    try:
        cmap = plt.colormaps['viridis']
    except:
        # Fallback for older matplotlib versions
        cmap = plt.cm.viridis
    
    # Define the layout grid
    grid_height = 100  # Total height
    row_height = grid_height / 2  # Each row has the same height
    
    # Calculate widths proportionally within each row
    row1_total = sum(row1_weights.values()) or 1  # Avoid division by zero
    row2_total = sum(row2_weights.values()) or 1

    prop_height = row1_total/(row1_total + row2_total)

    row_height1 = grid_height * prop_height
    row_height2 = grid_height * (1- prop_height)
    
    row1_positions = []
    row2_positions = []
    
    # Calculate positions for first row (digits 0-4)
    current_x = 0
    for digit in row1_digits:
        if digit in row1_weights and row1_weights[digit] > 0:
            # Calculate width proportionally based on weight
            width = 100 * (row1_weights[digit] / row1_total)
            row1_positions.append({
                'digit': digit,
                'x': current_x,
                'y': row_height2,  # Start from top row
                'width': width,
                'height': row_height1,
                'weight': row1_weights[digit]
            })
            current_x += width
    
    # Calculate positions for second row (digits 5-9)
    current_x = 0
    for digit in row2_digits:
        if digit in row2_weights and row2_weights[digit] > 0:
            # Calculate width proportionally based on weight
            width = 100 * (row2_weights[digit] / row2_total)
            row2_positions.append({
                'digit': digit,
                'x': current_x,
                'y': 0,  # Start from bottom row
                'width': width,
                'height': row_height2,
                'weight': row2_weights[digit]
            })
            current_x += width
    
    # Combine all positions
    all_positions = row1_positions + row2_positions
    
    # Now draw each digit category and its nested digits
    for pos in all_positions:
        digit = pos['digit']
        cat_x, cat_y = pos['x'], pos['y']
        cat_dx, cat_dy = pos['width'], pos['height']
        
        # Get color for this digit
        color = cmap(digit / 10)
        
        # Draw category background
        category_patch = patches.Rectangle(
            (cat_x, cat_y), cat_dx, cat_dy,
            linewidth=0,
            facecolor=color,
            alpha=0.2
        )
        ax.add_patch(category_patch)
        
        # Draw category border
        border_rect = patches.Rectangle(
            (cat_x, cat_y), cat_dx, cat_dy,
            linewidth=category_border_width,
            edgecolor='black',
            facecolor='none',
            zorder=5
        )
        ax.add_patch(border_rect)
        
        # Add total weight text at the top of the category
        # ax.text(
        #     cat_x + cat_dx/2,
        #     cat_y + cat_dy - 2,
        #     f"Total: {pos['weight']:.1f}",
        #     fontsize=10,
        #     ha='center',
        #     va='top',
        #     color='black',
        #     weight='bold',
        #     bbox=dict(facecolor='white', alpha=0.7),
        #     zorder=6
        # )
        
        # Get items for this digit
        if digit in digit_groups:
            items = digit_groups[digit]
            
            # Limit number of images if needed
            if len(items) > max_images_per_digit:
                # Sort by weight in descending order
                items.sort(key=lambda x: x[1], reverse=True)
                items = items[:max_images_per_digit]
            
            # Get indices and weights
            item_indices, item_weights = zip(*items) if items else ([], [])
            
            # Make sure there's at least 1 item
            if len(item_weights) > 0:
                # Use the full category space without padding
                padded_x = cat_x
                padded_y = cat_y
                padded_dx = cat_dx
                padded_dy = cat_dy
                
                # Only proceed if we have enough space
                if padded_dx > 0 and padded_dy > 0:
                    nested_norm = squarify.normalize_sizes(item_weights, padded_dx, padded_dy)
                    nested_rects = squarify.squarify(nested_norm, padded_x, padded_y, padded_dx, padded_dy)
                    
                    # Draw each nested rectangle with MNIST image
                    for j, n_rect in enumerate(nested_rects):
                        nx, ny, ndx, ndy = n_rect['x'], n_rect['y'], n_rect['dx'], n_rect['dy']
                        
                        # Skip if too small
                        if ndx < 1 or ndy < 1:
                            continue
                        
                        # Calculate image position with border
                        img_x = nx + image_border_width/2
                        img_y = ny + image_border_width/2
                        img_width = ndx - image_border_width
                        img_height = ndy - image_border_width
                        
                        # Draw border around image
                        img_border = patches.Rectangle(
                            (nx, ny), ndx, ndy,
                            linewidth=image_border_width,
                            edgecolor='white',
                            facecolor='none',
                            zorder=3
                        )
                        ax.add_patch(img_border)
                        
                        # Get MNIST image data
                        img_data = X[item_indices[j]].reshape(28, 28)
                        
                        # Draw image
                        ax.imshow(
                            img_data,
                            cmap='gray',
                            extent=[img_x, img_x+img_width, img_y, img_y+img_height],
                            aspect='auto',
                            zorder=2
                        )
                        
                        # Add weight text on top of larger images
                        # if ndx >= 5 and ndy >= 5:
                        #     ax.text(
                        #         nx + ndx/2, ny + ndy - 0.5,
                        #         f"{item_weights[j]:.1f}",
                        #         fontsize=min(8, max(6, ndx/4)),
                        #         ha='center',
                        #         va='top',
                        #         color='white',
                        #         bbox=dict(facecolor='black', alpha=0.5, pad=0.1),
                        #         zorder=4
                        #     )
    
    # Set up the axis
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove plot title and legend as requested
    # (No title or legend code here)
    
    plt.tight_layout()
    return fig

def mnist_treemap_example(num_samples=1000, weight_method='chisquared', 
                         max_images_per_digit=50,
                         image_border_width=0.2,
                         category_border_width=0.6):
    """Generate a two-row MNIST treemap example."""
    # Load data (1000 images = 100 per digit)
    X, y = load_mnist_subset(num_samples)
    print(f"Loaded {len(y)} MNIST digits")
    
    # Assign weights
    weights = assign_weights(y, method=weight_method)
    print(f"Assigned weights using method: {weight_method}")
    
    # Create two-row treemap
    fig = create_two_row_treemap(
        X, y, weights, 
        figsize=(16, 10), 
        max_images_per_digit=max_images_per_digit,
        image_border_width=image_border_width,
        category_border_width=category_border_width
    )
    
    plt.show()
    return fig

if __name__ == "__main__":
    # Create ultra-minimal treemap (no title, no legend)
    mnist_treemap_example(1000, 'chisquared', 1000)
