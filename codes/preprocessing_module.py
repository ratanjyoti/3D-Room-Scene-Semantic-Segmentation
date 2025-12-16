"""
Step 2: Point Cloud Preprocessing
Implements noise removal and downsampling for 3D point clouds.
"""

import numpy as np
from scipy.spatial import cKDTree


class PointCloudPreprocessor:
    """
    Preprocessing pipeline for 3D point clouds
    - Statistical outlier removal
    - Radius outlier removal
    - Voxel downsampling
    """
    
    def __init__(self):
        self.original_count = 0
        self.processed_count = 0
    
    def statistical_outlier_removal(self, points, k=20, std_ratio=2.0):
        """
        Remove statistical outliers based on distance to neighbors.
        
        Args:
            points: Nx3 numpy array of XYZ coordinates
            k: Number of nearest neighbors to consider
            std_ratio: Standard deviation multiplier threshold
            
        Returns:
            filtered_points: Points without outliers
            inlier_mask: Boolean mask of inliers
        """
        print(f"\n{'─' * 60}")
        print("Statistical Outlier Removal")
        print(f"{'─' * 60}")
        print(f"  Input points: {len(points):,}")
        print(f"  K neighbors: {k}")
        print(f"  Std ratio: {std_ratio}")
        
        # Build KD-tree for efficient neighbor search
        tree = cKDTree(points)
        
        # Find k nearest neighbors for each point
        distances, _ = tree.query(points, k=k+1)  # +1 because point itself is included
        
        # Calculate mean distance to neighbors (excluding self)
        mean_distances = distances[:, 1:].mean(axis=1)
        
        # Calculate threshold
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        threshold = global_mean + std_ratio * global_std
        
        # Filter outliers
        inlier_mask = mean_distances < threshold
        filtered_points = points[inlier_mask]
        
        removed = len(points) - len(filtered_points)
        print(f"  Removed: {removed:,} points ({removed/len(points)*100:.2f}%)")
        print(f"  Remaining: {len(filtered_points):,} points")
        
        return filtered_points, inlier_mask
    
    def radius_outlier_removal(self, points, radius=0.05, min_neighbors=10):
        """
        Remove points with few neighbors within radius.
        
        Args:
            points: Nx3 numpy array
            radius: Search radius
            min_neighbors: Minimum number of neighbors required
            
        Returns:
            filtered_points: Points without outliers
            inlier_mask: Boolean mask of inliers
        """
        print(f"\n{'─' * 60}")
        print("Radius Outlier Removal")
        print(f"{'─' * 60}")
        print(f"  Input points: {len(points):,}")
        print(f"  Radius: {radius}")
        print(f"  Min neighbors: {min_neighbors}")
        
        tree = cKDTree(points)
        
        # Count neighbors within radius for each point
        neighbor_counts = np.array([len(tree.query_ball_point(p, radius)) - 1 
                                   for p in points])
        
        # Filter points with sufficient neighbors
        inlier_mask = neighbor_counts >= min_neighbors
        filtered_points = points[inlier_mask]
        
        removed = len(points) - len(filtered_points)
        print(f"  Removed: {removed:,} points ({removed/len(points)*100:.2f}%)")
        print(f"  Remaining: {len(filtered_points):,} points")
        
        return filtered_points, inlier_mask
    
    def voxel_downsample(self, points, colors=None, voxel_size=0.02):
        """
        Downsample point cloud using voxel grid.
        
        Args:
            points: Nx3 numpy array
            colors: Nx3 numpy array (optional)
            voxel_size: Size of voxel grid
            
        Returns:
            downsampled_points: Downsampled points
            downsampled_colors: Downsampled colors (if provided)
        """
        print(f"\n{'─' * 60}")
        print("Voxel Downsampling")
        print(f"{'─' * 60}")
        print(f"  Input points: {len(points):,}")
        print(f"  Voxel size: {voxel_size}")
        
        # Compute voxel indices for each point
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Create unique voxel keys
        voxel_keys = {}
        
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            
            if key not in voxel_keys:
                voxel_keys[key] = {
                    'points': [points[i]],
                    'colors': [colors[i]] if colors is not None else []
                }
            else:
                voxel_keys[key]['points'].append(points[i])
                if colors is not None:
                    voxel_keys[key]['colors'].append(colors[i])
        
        # Average points and colors in each voxel
        downsampled_points = []
        downsampled_colors = []
        
        for voxel_data in voxel_keys.values():
            downsampled_points.append(np.mean(voxel_data['points'], axis=0))
            if colors is not None:
                downsampled_colors.append(np.mean(voxel_data['colors'], axis=0))
        
        downsampled_points = np.array(downsampled_points)
        downsampled_colors = np.array(downsampled_colors) if colors is not None else None
        
        reduction = (1 - len(downsampled_points) / len(points)) * 100
        print(f"  Output points: {len(downsampled_points):,}")
        print(f"  Reduction: {reduction:.2f}%")
        
        return downsampled_points, downsampled_colors
    
    def preprocess(self, points, colors=None, 
                   use_statistical=True, use_radius=False,
                   voxel_size=0.02, **kwargs):
        """
        Complete preprocessing pipeline.
        
        Args:
            points: Nx3 point cloud
            colors: Nx3 colors (optional)
            use_statistical: Apply statistical outlier removal
            use_radius: Apply radius outlier removal
            voxel_size: Voxel size for downsampling
            **kwargs: Additional parameters for outlier removal
            
        Returns:
            processed_points: Cleaned and downsampled points
            processed_colors: Corresponding colors (if provided)
        """
        print(f"\n{'=' * 60}")
        print("PREPROCESSING PIPELINE")
        print(f"{'=' * 60}")
        
        self.original_count = len(points)
        processed_points = points.copy()
        processed_colors = colors.copy() if colors is not None else None
        
        # Statistical outlier removal
        if use_statistical:
            filtered_points, mask = self.statistical_outlier_removal(
                processed_points,
                k=kwargs.get('stat_k', 20),
                std_ratio=kwargs.get('stat_std', 2.0)
            )
            processed_points = filtered_points
            if processed_colors is not None:
                processed_colors = processed_colors[mask]
        
        # Radius outlier removal
        if use_radius:
            filtered_points, mask = self.radius_outlier_removal(
                processed_points,
                radius=kwargs.get('radius', 0.05),
                min_neighbors=kwargs.get('min_neighbors', 10)
            )
            processed_points = filtered_points
            if processed_colors is not None:
                processed_colors = processed_colors[mask]
        
        # Voxel downsampling
        processed_points, processed_colors = self.voxel_downsample(
            processed_points, processed_colors, voxel_size
        )
        
        self.processed_count = len(processed_points)
        
        print(f"\n{'=' * 60}")
        print("PREPROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Original: {self.original_count:,} points")
        print(f"  Final: {self.processed_count:,} points")
        print(f"  Total reduction: {(1 - self.processed_count/self.original_count)*100:.2f}%")
        print(f"{'=' * 60}\n")
        
        return processed_points, processed_colors


# Example usage
if __name__ == "__main__":
    # Create sample point cloud
    np.random.seed(42)
    n_points = 10000
    
    # Generate room-like structure
    points = np.random.randn(n_points, 3)
    points[:, 2] = np.abs(points[:, 2])  # Keep Z positive
    
    # Add some outliers
    outliers = np.random.randn(100, 3) * 5
    points = np.vstack([points, outliers])
    
    colors = np.random.randint(0, 255, (len(points), 3))
    
    # Initialize preprocessor
    preprocessor = PointCloudPreprocessor()
    
    # Run preprocessing
    clean_points, clean_colors = preprocessor.preprocess(
        points, colors,
        use_statistical=True,
        use_radius=False,
        voxel_size=0.05,
        stat_k=20,
        stat_std=2.0
    )
    
    print("Preprocessing example completed!")