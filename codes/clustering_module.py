"""
Step 3: Point Cloud Clustering
Implements DBSCAN and Euclidean clustering for scene segmentation.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from collections import Counter


class PointCloudClusterer:
    """
    Clustering algorithms for 3D point cloud segmentation
    """
    
    def __init__(self):
        self.clusters = None
        self.cluster_info = []
    
    def dbscan_clustering(self, points, eps=0.05, min_samples=10):
        """
        DBSCAN clustering for point cloud segmentation.
        
        Args:
            points: Nx3 numpy array
            eps: Maximum distance between neighbors
            min_samples: Minimum points to form a cluster
            
        Returns:
            cluster_labels: Cluster ID for each point (-1 for noise)
            num_clusters: Number of clusters found
        """
        print(f"\n{'=' * 60}")
        print("DBSCAN CLUSTERING")
        print(f"{'=' * 60}")
        print(f"  Points: {len(points):,}")
        print(f"  Epsilon: {eps}")
        print(f"  Min samples: {min_samples}")
        
        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clustering.fit_predict(points)
        
        # Count clusters (excluding noise points labeled as -1)
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = np.sum(labels == -1)
        
        print(f"\n  Results:")
        print(f"    Clusters found: {num_clusters}")
        print(f"    Noise points: {noise_points:,} ({noise_points/len(points)*100:.2f}%)")
        
        # Cluster size distribution
        if num_clusters > 0:
            cluster_sizes = Counter(labels[labels >= 0])
            print(f"\n  Cluster sizes:")
            for cluster_id in sorted(cluster_sizes.keys())[:10]:  # Show first 10
                size = cluster_sizes[cluster_id]
                print(f"    Cluster {cluster_id}: {size:,} points")
            if num_clusters > 10:
                print(f"    ... and {num_clusters - 10} more clusters")
        
        self.clusters = labels
        return labels, num_clusters
    
    def euclidean_clustering(self, points, tolerance=0.05, min_cluster_size=50, max_cluster_size=None):
        """
        Euclidean clustering (region growing) for point cloud segmentation.
        
        Args:
            points: Nx3 numpy array
            tolerance: Distance threshold for clustering
            min_cluster_size: Minimum points per cluster
            max_cluster_size: Maximum points per cluster (None = no limit)
            
        Returns:
            cluster_labels: Cluster ID for each point (-1 for unassigned)
            num_clusters: Number of clusters found
        """
        print(f"\n{'=' * 60}")
        print("EUCLIDEAN CLUSTERING")
        print(f"{'=' * 60}")
        print(f"  Points: {len(points):,}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Min cluster size: {min_cluster_size}")
        
        # Build KD-tree for efficient neighbor search
        tree = cKDTree(points)
        
        # Initialize labels (-1 = unassigned)
        labels = np.full(len(points), -1, dtype=np.int32)
        cluster_id = 0
        
        for i in range(len(points)):
            if labels[i] != -1:
                continue  # Already assigned
            
            # Start new cluster with region growing
            queue = [i]
            labels[i] = cluster_id
            cluster_points = [i]
            
            while queue:
                current_idx = queue.pop(0)
                
                # Find neighbors within tolerance
                neighbors = tree.query_ball_point(points[current_idx], tolerance)
                
                for neighbor_idx in neighbors:
                    if labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_id
                        cluster_points.append(neighbor_idx)
                        queue.append(neighbor_idx)
                        
                        # Stop if max size reached
                        if max_cluster_size and len(cluster_points) >= max_cluster_size:
                            break
                
                if max_cluster_size and len(cluster_points) >= max_cluster_size:
                    break
            
            # Keep cluster only if it meets minimum size
            if len(cluster_points) >= min_cluster_size:
                cluster_id += 1
            else:
                # Mark as noise
                labels[cluster_points] = -1
        
        num_clusters = cluster_id
        noise_points = np.sum(labels == -1)
        
        print(f"\n  Results:")
        print(f"    Clusters found: {num_clusters}")
        print(f"    Noise points: {noise_points:,} ({noise_points/len(points)*100:.2f}%)")
        
        self.clusters = labels
        return labels, num_clusters
    
    def analyze_clusters(self, points, labels):
        """
        Analyze cluster properties for labeling.
        
        Args:
            points: Nx3 numpy array
            labels: Cluster labels for each point
            
        Returns:
            cluster_info: List of dictionaries with cluster properties
        """
        print(f"\n{'=' * 60}")
        print("CLUSTER ANALYSIS")
        print(f"{'=' * 60}")
        
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        cluster_info = []
        
        for label in sorted(unique_labels):
            mask = labels == label
            cluster_points = points[mask]
            
            # Calculate properties
            centroid = cluster_points.mean(axis=0)
            size = len(cluster_points)
            
            # Bounding box
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            dimensions = max_bound - min_bound
            
            # Calculate planarity (using PCA)
            centered = cluster_points - centroid
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
            
            # Planarity measure (higher = more planar)
            planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
            
            # Orientation (normal vector is smallest eigenvector)
            normal = eigenvectors[:, np.argmin(np.abs(eigenvalues))]
            
            # Check if horizontal or vertical
            vertical_threshold = 0.7
            is_horizontal = abs(normal[2]) > vertical_threshold
            is_vertical = abs(normal[2]) < (1 - vertical_threshold)
            
            info = {
                'id': label,
                'size': size,
                'centroid': centroid,
                'min_bound': min_bound,
                'max_bound': max_bound,
                'dimensions': dimensions,
                'planarity': planarity,
                'normal': normal,
                'is_horizontal': is_horizontal,
                'is_vertical': is_vertical,
                'height': centroid[2]
            }
            
            cluster_info.append(info)
        
        # Sort by size (largest first)
        cluster_info = sorted(cluster_info, key=lambda x: x['size'], reverse=True)
        
        print(f"\n  Top 10 Largest Clusters:")
        print(f"  {'ID':>4} {'Size':>8} {'Centroid (X,Y,Z)':>30} {'Planar':>7} {'Orient':>8}")
        print(f"  {'-'*70}")
        
        for info in cluster_info[:10]:
            orient = 'Horiz' if info['is_horizontal'] else 'Vert' if info['is_vertical'] else 'Other'
            print(f"  {info['id']:>4} {info['size']:>8,} "
                  f"({info['centroid'][0]:>6.2f},{info['centroid'][1]:>6.2f},{info['centroid'][2]:>6.2f}) "
                  f"{info['planarity']:>6.3f} {orient:>8}")
        
        self.cluster_info = cluster_info
        return cluster_info
    
    def filter_small_clusters(self, labels, min_size=100):
        """
        Remove clusters smaller than threshold.
        
        Args:
            labels: Cluster labels
            min_size: Minimum cluster size
            
        Returns:
            filtered_labels: Labels with small clusters removed
        """
        print(f"\n{'─' * 60}")
        print(f"Filtering clusters smaller than {min_size} points")
        
        filtered_labels = labels.copy()
        cluster_sizes = Counter(labels[labels >= 0])
        
        removed_count = 0
        for cluster_id, size in cluster_sizes.items():
            if size < min_size:
                filtered_labels[filtered_labels == cluster_id] = -1
                removed_count += 1
        
        print(f"  Removed {removed_count} small clusters")
        
        return filtered_labels


# Example usage
if __name__ == "__main__":
    # Create sample room-like point cloud
    np.random.seed(42)
    
    # Floor
    floor = np.random.rand(5000, 3) * [5, 5, 0.1]
    
    # Ceiling
    ceiling = np.random.rand(3000, 3) * [5, 5, 0.1] + [0, 0, 2.9]
    
    # Walls
    wall1 = np.random.rand(2000, 3) * [5, 0.1, 3] + [0, 0, 0]
    wall2 = np.random.rand(2000, 3) * [5, 0.1, 3] + [0, 4.9, 0]
    
    # Furniture
    table = np.random.rand(1000, 3) * [1, 1, 0.8] + [2, 2, 0]
    chair = np.random.rand(500, 3) * [0.5, 0.5, 1] + [1, 2, 0]
    
    # Combine
    points = np.vstack([floor, ceiling, wall1, wall2, table, chair])
    
    # Add some noise
    noise = np.random.rand(200, 3) * [5, 5, 3]
    points = np.vstack([points, noise])
    
    print(f"Created sample room with {len(points):,} points")
    
    # Initialize clusterer
    clusterer = PointCloudClusterer()
    
    # Run DBSCAN clustering
    labels, num_clusters = clusterer.dbscan_clustering(points, eps=0.1, min_samples=20)
    
    # Analyze clusters
    cluster_info = clusterer.analyze_clusters(points, labels)
    
    # Filter small clusters
    filtered_labels = clusterer.filter_small_clusters(labels, min_size=500)
    
    print("\nClustering example completed!")