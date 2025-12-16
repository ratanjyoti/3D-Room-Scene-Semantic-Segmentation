"""
Step 4: Rule-Based Semantic Labeling
Assigns semantic labels (floor, ceiling, walls, furniture) based on geometric rules.
"""

import numpy as np


class RuleBasedLabeler:
    """
    Rule-based semantic labeling for indoor scenes
    """
    
    # Label definitions
    LABELS = {
        'unknown': 0,
        'floor': 1,
        'ceiling': 2,
        'wall': 3,
        'furniture': 4,
        'noise': -1
    }
    
    LABEL_NAMES = {v: k for k, v in LABELS.items()}
    
    # Colors for visualization (RGB)
    COLORS = {
        'unknown': [128, 128, 128],   # Gray
        'floor': [102, 51, 0],         # Brown
        'ceiling': [200, 200, 255],    # Light blue
        'wall': [255, 200, 100],       # Light orange
        'furniture': [100, 200, 100],  # Green
        'noise': [50, 50, 50]          # Dark gray
    }
    
    def __init__(self):
        self.semantic_labels = None
        self.label_stats = {}
    
    def label_clusters(self, points, cluster_labels, cluster_info, 
                       floor_height_ratio=0.15, ceiling_height_ratio=0.85,
                       planarity_threshold=0.5, min_wall_height=1.0):
        """
        Assign semantic labels to clusters based on geometric rules.
        
        Args:
            points: Nx3 point cloud
            cluster_labels: Cluster ID for each point
            cluster_info: List of cluster properties from analyzer
            floor_height_ratio: Height ratio threshold for floor
            ceiling_height_ratio: Height ratio threshold for ceiling
            planarity_threshold: Minimum planarity for surfaces
            min_wall_height: Minimum height for wall classification
            
        Returns:
            semantic_labels: Semantic label for each point
            labeled_clusters: Dictionary mapping cluster ID to semantic label
        """
        print(f"\n{'=' * 60}")
        print("RULE-BASED SEMANTIC LABELING")
        print(f"{'=' * 60}")
        
        # Initialize semantic labels as unknown
        semantic_labels = np.full(len(points), self.LABELS['unknown'], dtype=np.int32)
        
        # Mark noise points
        semantic_labels[cluster_labels == -1] = self.LABELS['noise']
        
        # Get scene bounds for height-based rules
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        z_range = z_max - z_min
        
        print(f"\n  Scene vertical bounds:")
        print(f"    Z min: {z_min:.2f}")
        print(f"    Z max: {z_max:.2f}")
        print(f"    Z range: {z_range:.2f}")
        
        # Track labeled clusters
        labeled_clusters = {}
        
        # Sort clusters by size (process largest first)
        sorted_clusters = sorted(cluster_info, key=lambda x: x['size'], reverse=True)
        
        # Find floor candidate (lowest large horizontal planar surface)
        floor_candidates = [
            c for c in sorted_clusters
            if c['is_horizontal'] and 
               c['planarity'] > planarity_threshold and
               c['height'] < z_min + z_range * floor_height_ratio
        ]
        
        if floor_candidates:
            floor_cluster = floor_candidates[0]  # Largest candidate
            mask = cluster_labels == floor_cluster['id']
            semantic_labels[mask] = self.LABELS['floor']
            labeled_clusters[floor_cluster['id']] = 'floor'
            print(f"\n  ✓ Floor: Cluster {floor_cluster['id']} ({floor_cluster['size']:,} points)")
        
        # Find ceiling candidate (highest large horizontal planar surface)
        ceiling_candidates = [
            c for c in sorted_clusters
            if c['is_horizontal'] and 
               c['planarity'] > planarity_threshold and
               c['height'] > z_min + z_range * ceiling_height_ratio and
               c['id'] not in labeled_clusters
        ]
        
        if ceiling_candidates:
            ceiling_cluster = ceiling_candidates[0]
            mask = cluster_labels == ceiling_cluster['id']
            semantic_labels[mask] = self.LABELS['ceiling']
            labeled_clusters[ceiling_cluster['id']] = 'ceiling'
            print(f"  ✓ Ceiling: Cluster {ceiling_cluster['id']} ({ceiling_cluster['size']:,} points)")
        
        # Find walls (large vertical planar surfaces)
        wall_candidates = [
            c for c in sorted_clusters
            if c['is_vertical'] and 
               c['planarity'] > planarity_threshold and
               c['dimensions'][2] > min_wall_height and
               c['id'] not in labeled_clusters
        ]
        
        print(f"  ✓ Walls: {len(wall_candidates)} clusters")
        for wall_cluster in wall_candidates:
            mask = cluster_labels == wall_cluster['id']
            semantic_labels[mask] = self.LABELS['wall']
            labeled_clusters[wall_cluster['id']] = 'wall'
            print(f"      Cluster {wall_cluster['id']} ({wall_cluster['size']:,} points)")
        
        # Remaining clusters are furniture
        furniture_clusters = [
            c for c in sorted_clusters
            if c['id'] not in labeled_clusters
        ]
        
        print(f"  ✓ Furniture: {len(furniture_clusters)} clusters")
        for furniture_cluster in furniture_clusters[:10]:  # Show first 10
            mask = cluster_labels == furniture_cluster['id']
            semantic_labels[mask] = self.LABELS['furniture']
            labeled_clusters[furniture_cluster['id']] = 'furniture'
            print(f"      Cluster {furniture_cluster['id']} ({furniture_cluster['size']:,} points)")
        
        # Label remaining (if any)
        for furniture_cluster in furniture_clusters[10:]:
            mask = cluster_labels == furniture_cluster['id']
            semantic_labels[mask] = self.LABELS['furniture']
            labeled_clusters[furniture_cluster['id']] = 'furniture'
        
        if len(furniture_clusters) > 10:
            print(f"      ... and {len(furniture_clusters) - 10} more")
        
        # Calculate statistics
        self._calculate_label_stats(semantic_labels)
        
        self.semantic_labels = semantic_labels
        return semantic_labels, labeled_clusters
    
    def _calculate_label_stats(self, semantic_labels):
        """Calculate and display labeling statistics."""
        print(f"\n{'─' * 60}")
        print("LABELING STATISTICS")
        print(f"{'─' * 60}")
        
        total_points = len(semantic_labels)
        self.label_stats = {}
        
        for label_value, label_name in self.LABEL_NAMES.items():
            count = np.sum(semantic_labels == label_value)
            percentage = count / total_points * 100
            self.label_stats[label_name] = {
                'count': count,
                'percentage': percentage
            }
            print(f"  {label_name.capitalize():12s}: {count:8,} points ({percentage:5.2f}%)")
        
        print(f"{'─' * 60}")
    
    def get_colored_points(self, semantic_labels):
        """
        Generate colors for points based on semantic labels.
        
        Args:
            semantic_labels: Semantic label for each point
            
        Returns:
            colors: Nx3 RGB colors
        """
        colors = np.zeros((len(semantic_labels), 3), dtype=np.uint8)
        
        for label_value, label_name in self.LABEL_NAMES.items():
            mask = semantic_labels == label_value
            colors[mask] = self.COLORS[label_name]
        
        return colors
    
    def refine_labels(self, points, semantic_labels, cluster_labels):
        """
        Refine labels using neighborhood voting.
        
        Args:
            points: Nx3 point cloud
            semantic_labels: Current semantic labels
            cluster_labels: Cluster assignments
            
        Returns:
            refined_labels: Refined semantic labels
        """
        print(f"\n{'─' * 60}")
        print("REFINING LABELS")
        print(f"{'─' * 60}")
        
        from scipy.spatial import cKDTree
        
        refined_labels = semantic_labels.copy()
        tree = cKDTree(points)
        
        # Only refine unknown and furniture points
        uncertain_mask = (semantic_labels == self.LABELS['unknown']) | \
                         (semantic_labels == self.LABELS['furniture'])
        uncertain_indices = np.where(uncertain_mask)[0]
        
        changed_count = 0
        
        for idx in uncertain_indices:
            # Find neighbors
            neighbors = tree.query_ball_point(points[idx], r=0.1)
            neighbor_labels = semantic_labels[neighbors]
            
            # Vote (exclude unknown and noise)
            valid_labels = neighbor_labels[
                (neighbor_labels != self.LABELS['unknown']) & 
                (neighbor_labels != self.LABELS['noise'])
            ]
            
            if len(valid_labels) > 5:  # Need sufficient neighbors
                # Most common label
                from collections import Counter
                label_counts = Counter(valid_labels)
                most_common_label, count = label_counts.most_common(1)[0]
                
                # Update if strong consensus (>60% agreement)
                if count / len(valid_labels) > 0.6:
                    refined_labels[idx] = most_common_label
                    changed_count += 1
        
        print(f"  Refined {changed_count:,} labels")
        
        return refined_labels


# Example usage
if __name__ == "__main__":
    # Create sample clustered room
    np.random.seed(42)
    
    # Floor cluster (ID=0)
    floor = np.random.rand(5000, 3) * [5, 5, 0.1]
    floor_labels = np.zeros(5000, dtype=int)
    
    # Ceiling cluster (ID=1)
    ceiling = np.random.rand(3000, 3) * [5, 5, 0.1] + [0, 0, 2.9]
    ceiling_labels = np.ones(3000, dtype=int)
    
    # Wall clusters (ID=2,3)
    wall1 = np.random.rand(2000, 3) * [5, 0.1, 3] + [0, 0, 0]
    wall1_labels = np.full(2000, 2, dtype=int)
    
    wall2 = np.random.rand(2000, 3) * [5, 0.1, 3] + [0, 4.9, 0]
    wall2_labels = np.full(2000, 3, dtype=int)
    
    # Furniture clusters (ID=4,5)
    table = np.random.rand(1000, 3) * [1, 1, 0.8] + [2, 2, 0]
    table_labels = np.full(1000, 4, dtype=int)
    
    chair = np.random.rand(500, 3) * [0.5, 0.5, 1] + [1, 2, 0]
    chair_labels = np.full(500, 5, dtype=int)
    
    # Combine
    points = np.vstack([floor, ceiling, wall1, wall2, table, chair])
    cluster_labels = np.concatenate([floor_labels, ceiling_labels, 
                                      wall1_labels, wall2_labels,
                                      table_labels, chair_labels])
    
    # Create mock cluster info
    cluster_info = [
        {'id': 0, 'size': 5000, 'centroid': [2.5, 2.5, 0.05], 'height': 0.05,
         'is_horizontal': True, 'is_vertical': False, 'planarity': 0.9,
         'dimensions': [5, 5, 0.1]},
        {'id': 1, 'size': 3000, 'centroid': [2.5, 2.5, 2.95], 'height': 2.95,
         'is_horizontal': True, 'is_vertical': False, 'planarity': 0.9,
         'dimensions': [5, 5, 0.1]},
        {'id': 2, 'size': 2000, 'centroid': [2.5, 0.05, 1.5], 'height': 1.5,
         'is_horizontal': False, 'is_vertical': True, 'planarity': 0.8,
         'dimensions': [5, 0.1, 3]},
        {'id': 3, 'size': 2000, 'centroid': [2.5, 4.95, 1.5], 'height': 1.5,
         'is_horizontal': False, 'is_vertical': True, 'planarity': 0.8,
         'dimensions': [5, 0.1, 3]},
        {'id': 4, 'size': 1000, 'centroid': [2.5, 2.5, 0.4], 'height': 0.4,
         'is_horizontal': False, 'is_vertical': False, 'planarity': 0.5,
         'dimensions': [1, 1, 0.8]},
        {'id': 5, 'size': 500, 'centroid': [1.25, 2.25, 0.5], 'height': 0.5,
         'is_horizontal': False, 'is_vertical': False, 'planarity': 0.4,
         'dimensions': [0.5, 0.5, 1]},
    ]
    
    # Initialize labeler
    labeler = RuleBasedLabeler()
    
    # Apply rule-based labeling
    semantic_labels, labeled_clusters = labeler.label_clusters(
        points, cluster_labels, cluster_info
    )
    
    # Get colors
    colors = labeler.get_colored_points(semantic_labels)
    
    print("\n✓ Rule-based labeling example completed!")