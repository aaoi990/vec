import tmap as tm
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform


def farthest_point_first(points, k=5, initial_idx=None):
    """
    Farthest-Point-First traversal to select k diverse points
    
    Args:
        points: Array of points (n x d)
        k: Number of points to select
        initial_idx: Index of starting point (random if None)
        
    Returns:
        Indices of selected points
    """
    n = len(points)
    if k > n:
        return list(range(n))
    
    # Calculate pairwise distances between all points
    dist_matrix = squareform(pdist(points))
    
    # Select first point (random or specified)
    selected = []
    if initial_idx is None:
        initial_idx = np.random.randint(0, n)
    selected.append(initial_idx)
    
    # Select remaining points
    while len(selected) < k:
        # Find distances from each unselected point to all selected points
        mask = np.ones(n, dtype=bool)
        mask[selected] = False
        
        # For each unselected point, find minimum distance to any selected point
        unselected = np.where(mask)[0]
        min_distances = np.min([dist_matrix[s, unselected] for s in selected], axis=0)
        
        # Choose the point with maximum minimum distance
        farthest_idx = unselected[np.argmax(min_distances)]
        selected.append(farthest_idx)
    
    return selected


def main():
    """ Main function """
    n = 25
    edge_list = []

    # Create a random graph
    for i in range(n):
        for j in np.random.randint(0, high=n, size=2):
            edge_list.append([i, j, np.random.rand(1)])

    # Compute the layout
    x, y, s, t, _ = tm.layout_from_edge_list(
        n, edge_list, create_mst=False
    )
    
    # Convert tmap VectorFloat to numpy arrays
    x_array = np.array([x[i] for i in range(len(x))])
    y_array = np.array([y[i] for i in range(len(y))])
    
    # Combine x and y into points array
    points = np.column_stack((x_array, y_array))
    
    # Select 5 most diverse points using FPF
    selected_indices = farthest_point_first(points, k=5)
    
    # Plot the edges
    for i in range(len(s)):
        plt.plot(
            [x[s[i]], x[t[i]]],
            [y[s[i]], y[t[i]]],
            "k-",
            linewidth=0.5,
            alpha=0.5,
            zorder=1,
        )

    # Plot all vertices
    plt.scatter(x_array, y_array, c='blue', s=30, zorder=2, label='All Points')
    
    # Highlight the selected diverse points
    plt.scatter(
        [x_array[i] for i in selected_indices],
        [y_array[i] for i in selected_indices],
        c='red', 
        s=100, 
        zorder=3, 
        label='Most Diverse Points'
    )
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("diverse_points_fpf.png")


if __name__ == "__main__":
    main()
