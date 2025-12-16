"""
COMPLETE 3D ROOM SEGMENTATION PIPELINE
Integrates all modules: loading, preprocessing, clustering, labeling, and visualization
"""


import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add current directory to Python path (so it can find other modules)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all required modules
from preprocessing_module import PointCloudPreprocessor
from clustering_module import PointCloudClusterer
from labelling_module import RuleBasedLabeler
from visualization_module import PointCloudVisualizer, export_ply


class RoomSegmentationPipeline:
    """
    End-to-end pipeline for 3D room semantic segmentation
    """
    
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize modules
        self.preprocessor = None
        self.clusterer = None
        self.labeler = None
        self.visualizer = None
        
        # Data containers
        self.original_data = None
        self.processed_points = None
        self.processed_colors = None
        self.cluster_labels = None
        self.semantic_labels = None
    
    def load_data(self, anno_path):
        """
        Load point cloud from S3DIS Annotations folder.
        
        Args:
            anno_path: Path to Annotations folder
            
        Returns:
            success: Boolean indicating success
        """
        print(f"\n{'='*70}")
        print("STEP 1: DATA LOADING")
        print(f"{'='*70}")
        
        if not os.path.exists(anno_path):
            print(f"❌ Error: Path not found: {anno_path}")
            return False
        
        # Get all object files
        object_files = sorted([f for f in os.listdir(anno_path) if f.endswith('.txt')])
        print(f"Found {len(object_files)} object files\n")
        
        all_points = []
        all_colors = []
        
        for idx, obj_file in enumerate(object_files):
            obj_path = os.path.join(anno_path, obj_file)
            object_name = obj_file.replace('.txt', '')
            
            try:
                data = np.loadtxt(obj_path)
                
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                
                num_points = data.shape[0]
                points = data[:, :3]
                colors = data[:, 3:6] if data.shape[1] >= 6 else np.ones((num_points, 3)) * 128
                
                all_points.append(points)
                all_colors.append(colors)
                
                print(f"  {idx+1:2d}. {object_name:35s} {num_points:8,} points")
                
            except Exception as e:
                print(f"  ⚠️  Error loading {obj_file}: {str(e)}")
        
        if all_points:
            points = np.vstack(all_points)
            colors = np.vstack(all_colors)
            
            self.original_data = {
                'points': points,
                'colors': colors,
                'object_files': object_files
            }
            
            print(f"\n{'─'*70}")
            print(f"✓ Loaded {len(points):,} points from {len(object_files)} objects")
            print(f"  Bounds: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}] "
                  f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}] "
                  f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
            print(f"{'─'*70}")
            return True
        
        print("❌ No data loaded")
        return False
    
    def preprocess(self, voxel_size=0.02, use_statistical=True, use_radius=False):
        """
        Preprocess point cloud: noise removal and downsampling.
        
        Args:
            voxel_size: Voxel grid size for downsampling
            use_statistical: Apply statistical outlier removal
            use_radius: Apply radius outlier removal
            
        Returns:
            success: Boolean indicating success
        """
        print(f"\n{'='*70}")
        print("STEP 2: PREPROCESSING")
        print(f"{'='*70}")
        
        if self.original_data is None:
            print("❌ No data loaded. Run load_data() first.")
            return False
        
        self.preprocessor = PointCloudPreprocessor()
        
        self.processed_points, self.processed_colors = self.preprocessor.preprocess(
            self.original_data['points'],
            self.original_data['colors'],
            use_statistical=use_statistical,
            use_radius=use_radius,
            voxel_size=voxel_size,
            stat_k=20,
            stat_std=2.0
        )
        
        return True
    
    def cluster(self, method='dbscan', eps=0.05, min_samples=10, 
                min_cluster_size=100):
        """
        Cluster the preprocessed point cloud.
        
        Args:
            method: 'dbscan' or 'euclidean'
            eps: Distance threshold for clustering
            min_samples: Minimum samples for DBSCAN
            min_cluster_size: Minimum cluster size to keep
            
        Returns:
            success: Boolean indicating success
        """
        print(f"\n{'='*70}")
        print("STEP 3: CLUSTERING")
        print(f"{'='*70}")
        
        if self.processed_points is None:
            print("❌ No preprocessed data. Run preprocess() first.")
            return False
        
        self.clusterer = PointCloudClusterer()
        
        # Perform clustering
        if method == 'dbscan':
            self.cluster_labels, num_clusters = self.clusterer.dbscan_clustering(
                self.processed_points, eps=eps, min_samples=min_samples
            )
        elif method == 'euclidean':
            self.cluster_labels, num_clusters = self.clusterer.euclidean_clustering(
                self.processed_points, tolerance=eps, min_cluster_size=min_samples
            )
        else:
            print(f"❌ Unknown clustering method: {method}")
            return False
        
        # Analyze clusters
        self.cluster_info = self.clusterer.analyze_clusters(
            self.processed_points, self.cluster_labels
        )
        
        # Filter small clusters
        self.cluster_labels = self.clusterer.filter_small_clusters(
            self.cluster_labels, min_size=min_cluster_size
        )
        
        return True
    
    def label(self, floor_ratio=0.15, ceiling_ratio=0.85, 
              planarity_threshold=0.5, min_wall_height=1.0):
        """
        Apply rule-based semantic labeling.
        
        Args:
            floor_ratio: Height ratio threshold for floor detection
            ceiling_ratio: Height ratio threshold for ceiling detection
            planarity_threshold: Minimum planarity for surfaces
            min_wall_height: Minimum height for wall classification
            
        Returns:
            success: Boolean indicating success
        """
        print(f"\n{'='*70}")
        print("STEP 4: SEMANTIC LABELING")
        print(f"{'='*70}")
        
        if self.cluster_labels is None:
            print("❌ No clusters found. Run cluster() first.")
            return False
        
        self.labeler = RuleBasedLabeler()
        
        self.semantic_labels, labeled_clusters = self.labeler.label_clusters(
            self.processed_points,
            self.cluster_labels,
            self.cluster_info,
            floor_height_ratio=floor_ratio,
            ceiling_height_ratio=ceiling_ratio,
            planarity_threshold=planarity_threshold,
            min_wall_height=min_wall_height
        )
        
        return True
    
    def visualize_and_export(self, room_name="room", show_plots=True):
        """
        Visualize results and export to PLY.
        
        Args:
            room_name: Name for output files
            show_plots: Whether to display plots
            
        Returns:
            success: Boolean indicating success
        """
        print(f"\n{'='*70}")
        print("STEP 5: VISUALIZATION & EXPORT")
        print(f"{'='*70}")
        
        if self.semantic_labels is None:
            print("❌ No semantic labels. Run label() first.")
            return False
        
        self.visualizer = PointCloudVisualizer()
        
        # Get semantic colors
        semantic_colors = self.labeler.get_colored_points(self.semantic_labels)
        
        # 1. Original point cloud visualization
        print("\n1. Visualizing original point cloud...")
        self.visualizer.visualize_3d(
            self.processed_points,
            self.processed_colors,
            title=f"{room_name} - Original",
            save_path=os.path.join(self.output_dir, f"{room_name}_original.png"),
            show=show_plots
        )
        
        # 2. Segmented point cloud visualization
        print("\n2. Visualizing segmented point cloud...")
        self.visualizer.visualize_3d(
            self.processed_points,
            semantic_colors,
            title=f"{room_name} - Semantic Segmentation",
            save_path=os.path.join(self.output_dir, f"{room_name}_segmented.png"),
            show=show_plots
        )
        
        # 3. Top-down view
        print("\n3. Creating top-down view...")
        self.visualizer.visualize_2d_topdown(
            self.processed_points,
            semantic_colors,
            title=f"{room_name} - Floor Plan",
            save_path=os.path.join(self.output_dir, f"{room_name}_topdown.png"),
            show=show_plots
        )
        
        # 4. Comparison plot
        print("\n4. Creating comparison plot...")
        self.visualizer.create_comparison_plot(
            self.processed_points,
            self.processed_colors,
            semantic_colors,
            save_path=os.path.join(self.output_dir, f"{room_name}_comparison.png"),
            show=show_plots
        )
        
        # 5. Export PLY file
        print("\n5. Exporting PLY file...")
        export_ply(
            self.processed_points,
            semantic_colors,
            os.path.join(self.output_dir, f"{room_name}_segmented.ply")
        )
        
        print(f"\n{'='*70}")
        print(f"✓ All outputs saved to: {self.output_dir}/")
        print(f"{'='*70}")
        
        return True
    
    def run_complete_pipeline(self, anno_path, room_name="room",
                             voxel_size=0.02, eps=0.05, show_plots=True):
        """
        Run the complete segmentation pipeline.
        
        Args:
            anno_path: Path to Annotations folder
            room_name: Name for output files
            voxel_size: Voxel size for downsampling
            eps: Clustering distance threshold
            show_plots: Whether to display plots
            
        Returns:
            success: Boolean indicating success
        """
        print(f"\n{'#'*70}")
        print(f"# 3D ROOM SEMANTIC SEGMENTATION PIPELINE")
        print(f"# Room: {room_name}")
        print(f"{'#'*70}")
        
        # Step 1: Load data
        if not self.load_data(anno_path):
            return False
        
        # Step 2: Preprocess
        if not self.preprocess(voxel_size=voxel_size, use_statistical=True):
            return False
        
        # Step 3: Cluster
        if not self.cluster(method='dbscan', eps=eps, min_samples=10, 
                           min_cluster_size=100):
            return False
        
        # Step 4: Label
        if not self.label(floor_ratio=0.15, ceiling_ratio=0.85,
                         planarity_threshold=0.5, min_wall_height=1.0):
            return False
        
        # Step 5: Visualize and export
        if not self.visualize_and_export(room_name=room_name, show_plots=show_plots):
            return False
        
        print(f"\n{'#'*70}")
        print(f"# PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"# Check output folder: {self.output_dir}/")
        print(f"{'#'*70}\n")
        
        return True


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # Configuration
    DATASET_PATH = r"E:\3D-semantic\3D-Room-Scene-Semantic-Segmentation\dataset\Stanford3dDataset_v1.2_Aligned_Version"
    
    # Example: Process Area_1/conferenceRoom_1
    AREA = "Area_1"
    ROOM = "conferenceRoom_1"
    
    anno_path = os.path.join(DATASET_PATH, AREA, ROOM, "Annotations")
    
    print(f"\n{'='*70}")
    print(f"Processing: {AREA}/{ROOM}")
    print(f"Annotations path: {anno_path}")
    print(f"{'='*70}\n")
    
    # Check if path exists
    if not os.path.exists(anno_path):
        print(f"❌ Error: Annotations folder not found at {anno_path}")
        print("\nPlease update DATASET_PATH, AREA, and ROOM variables")
        print("to match your uploaded dataset structure.")
        exit(1)
    
    # Initialize pipeline
    pipeline = RoomSegmentationPipeline(output_dir=f"output_{AREA}_{ROOM}")
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline(
        anno_path=anno_path,
        room_name=f"{AREA}_{ROOM}",
        voxel_size=0.02,  # Adjust for density (smaller = more detail)
        eps=0.05,         # Adjust for clustering (smaller = more clusters)
        show_plots=False  # Set to True to display plots interactively
    )
    
    if success:
        print("\n✓ Segmentation completed successfully!")
        print(f"\nOutput files:")
        print(f"  - {AREA}_{ROOM}_original.png")
        print(f"  - {AREA}_{ROOM}_segmented.png")
        print(f"  - {AREA}_{ROOM}_topdown.png")
        print(f"  - {AREA}_{ROOM}_comparison.png")
        print(f"  - {AREA}_{ROOM}_segmented.ply")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")