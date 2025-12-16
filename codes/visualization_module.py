"""
Step 5: Point Cloud Visualization
Visualizes segmented point clouds and exports results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class PointCloudVisualizer:
    """
    Visualization tools for 3D point clouds
    """
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
    
    def visualize_3d(self, points, colors=None, title="3D Point Cloud", 
                     point_size=1, save_path=None, show=True):
        """
        Visualize point cloud in 3D.
        
        Args:
            points: Nx3 numpy array
            colors: Nx3 RGB colors (0-255) or None for default
            title: Plot title
            point_size: Size of points
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare colors
        if colors is not None:
            colors_normalized = colors / 255.0  # Normalize to 0-1
        else:
            colors_normalized = 'blue'
        
        # Downsample for visualization if too many points
        if len(points) > 50000:
            indices = np.random.choice(len(points), 50000, replace=False)
            points_vis = points[indices]
            if colors is not None:
                colors_vis = colors_normalized[indices]
            else:
                colors_vis = colors_normalized
            print(f"  Downsampled to 50,000 points for visualization")
        else:
            points_vis = points
            colors_vis = colors_normalized
        
        # Plot
        ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                  c=colors_vis, s=point_size, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_2d_topdown(self, points, colors=None, title="Top-Down View",
                            point_size=1, save_path=None, show=True):
        """
        Create 2D top-down projection (floor plan view).
        
        Args:
            points: Nx3 numpy array
            colors: Nx3 RGB colors
            title: Plot title
            point_size: Size of points
            save_path: Path to save figure
            show: Whether to display
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Prepare colors
        if colors is not None:
            colors_normalized = colors / 255.0
        else:
            colors_normalized = 'blue'
        
        # Plot X-Y projection
        ax.scatter(points[:, 0], points[:, 1], 
                  c=colors_normalized, s=point_size, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_clusters_grid(self, points, cluster_labels, max_clusters=12,
                               save_path=None, show=True):
        """
        Visualize individual clusters in a grid layout.
        
        Args:
            points: Nx3 numpy array
            cluster_labels: Cluster ID for each point
            max_clusters: Maximum clusters to display
            save_path: Path to save figure
            show: Whether to display
        """
        unique_labels = sorted(set(cluster_labels))
        unique_labels = [l for l in unique_labels if l >= 0][:max_clusters]
        
        n_clusters = len(unique_labels)
        cols = 4
        rows = (n_clusters + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, 5 * rows))
        
        for idx, label in enumerate(unique_labels):
            mask = cluster_labels == label
            cluster_points = points[mask]
            
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            
            # Random color for cluster
            color = np.random.rand(3)
            
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      cluster_points[:, 2], c=[color], s=1, alpha=0.6)
            
            ax.set_title(f'Cluster {label} ({len(cluster_points)} pts)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_semantic_labels(self, points, semantic_labels, label_colors,
                                  title="Semantic Segmentation", 
                                  save_path=None, show=True):
        """
        Visualize semantically labeled point cloud.
        
        Args:
            points: Nx3 numpy array
            semantic_labels: Semantic label for each point
            label_colors: Dictionary mapping label names to RGB colors
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        """
        # Create colors array
        from labelling_module import RuleBasedLabeler
        labeler = RuleBasedLabeler()
        colors = labeler.get_colored_points(semantic_labels)
        
        # Create 3D visualization
        self.visualize_3d(points, colors, title=title, 
                         save_path=save_path, show=show)
    
    def create_comparison_plot(self, points, original_colors, semantic_colors,
                              save_path=None, show=True):
        """
        Side-by-side comparison of original and segmented point cloud.
        
        Args:
            points: Nx3 numpy array
            original_colors: Original RGB colors
            semantic_colors: Semantic segmentation colors
            save_path: Path to save figure
            show: Whether to display
        """
        fig = plt.figure(figsize=(20, 8))
        
        # Original
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Downsample if needed
        if len(points) > 30000:
            indices = np.random.choice(len(points), 30000, replace=False)
            points_vis = points[indices]
            orig_vis = original_colors[indices] / 255.0
            sem_vis = semantic_colors[indices] / 255.0
        else:
            points_vis = points
            orig_vis = original_colors / 255.0
            sem_vis = semantic_colors / 255.0
        
        ax1.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                   c=orig_vis, s=1, alpha=0.6)
        ax1.set_title('Original Point Cloud')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Segmented
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                   c=sem_vis, s=1, alpha=0.6)
        ax2.set_title('Semantic Segmentation')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved comparison to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


import os
def export_ply(points, colors, output_path):
    """
    Export point cloud to PLY format.
    
    Args:
        points: Nx3 numpy array
        colors: Nx3 RGB colors (0-255)
        output_path: Path to save PLY file
    """
    print(f"\n{'─' * 60}")
    print(f"Exporting to PLY: {output_path}")
    print(f"{'─' * 60}")
    
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
    
    print(f"  ✓ Exported {len(points):,} points")
    print(f"  File size: {np.round(os.path.getsize(output_path) / 1024 / 1024, 2)} MB")



# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    n_points = 10000
    points = np.random.rand(n_points, 3) * 5
    colors = np.random.randint(0, 255, (n_points, 3))
    
    # Initialize visualizer
    visualizer = PointCloudVisualizer()
    
    # 3D visualization
    print("Creating 3D visualization...")
    visualizer.visualize_3d(points, colors, title="Sample Point Cloud")
    
    # Top-down view
    print("\nCreating top-down view...")
    visualizer.visualize_2d_topdown(points, colors, title="Floor Plan")
    
    # Export PLY
    print("\nExporting PLY file...")
    export_ply(points, colors, "sample_output.ply")
    
    print("\n✓ Visualization examples completed!")




# import numpy as np
# import plotly.graph_objects as go
# import numpy as np
# import plotly.graph_objects as go


# class PointCloudVisualizer:
#     def visualize_3d(self, points, colors=None, title="3D Point Cloud", point_size=2, save_path=None, show=True):
#         if colors is not None:
#             colors_hex = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in colors]
#         else:
#             colors_hex = 'blue'
#         fig = go.Figure(data=[go.Scatter3d(
#             x=points[:, 0],
#             y=points[:, 1],
#             z=points[:, 2],
#             mode='markers',
#             marker=dict(size=point_size, color=colors_hex, opacity=0.7)
#         )])
#         fig.update_layout(title=title, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
#         if save_path:
#             fig.write_html(save_path)
#         if show:
#             fig.show()


# def export_ply(points, colors, output_path):
#     with open(output_path, 'w') as f:
#         f.write("ply\nformat ascii 1.0\n")
#         f.write(f"element vertex {len(points)}\n")
#         f.write("property float x\nproperty float y\nproperty float z\n")
#         f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
#         f.write("end_header\n")
#         for i in range(len(points)):
#             f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
#             f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
# import plotly.graph_objects as go



# def visualize_2d_topdown(self, points, colors=None, title="Top-Down View",
#                              point_size=2, save_path=None, show=True):
#         """
#         Create 2D top-down projection (floor plan view) using Plotly.
#         Args:
#             points: Nx3 numpy array
#             colors: Nx3 RGB colors (0-255) or None for default
#             title: Plot title
#             point_size: Size of points
#             save_path: Path to save HTML file (optional)
#             show: Whether to display
#         """
#         if colors is not None:
#             colors_hex = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in colors]
#         else:
#             colors_hex = 'blue'


#         fig = go.Figure(data=[go.Scattergl(
#             x=points[:, 0],
#             y=points[:, 1],
#             mode='markers',
#             marker=dict(
#                 size=point_size,
#                 color=colors_hex,
#                 opacity=0.7
#             )
#         )])


#         fig.update_layout(
#             title=title,
#             xaxis_title='X',
#             yaxis_title='Y',
#             yaxis=dict(scaleanchor="x", scaleratio=1),
#             margin=dict(l=0, r=0, b=0, t=40)
#         )


#         if save_path:
#             fig.write_html(save_path)
#             print(f"  Saved top-down view to: {save_path}")


#         if show:
#             fig.show()


# if __name__ == "__main__":
#     # Create sample data
#     np.random.seed(42)
#     n_points = 10000
#     points = np.random.rand(n_points, 3) * 5
#     colors = np.random.randint(0, 255, (n_points, 3))


#     # 3D visualization with Plotly
#     print("Creating 3D visualization (Plotly)...")
#     visualize_3d_plotly(points, colors, title="Sample Point Cloud", save_path="sample_output.html", show=True)


#     # Export PLY
#     print("\nExporting PLY file...")
#     export_ply(points, colors, "sample_output.ply")


#     print("\n✓ Visualization and export completed!")

# is this correct code