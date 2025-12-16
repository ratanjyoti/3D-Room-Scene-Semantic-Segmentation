"""
SIMPLE TEST SCRIPT WITH VISUALIZATION
Test the trained model on a new room and generate preview images!
Prerequisites: 
- trained_model.pkl must exist (run simple_train.py first)
- matplotlib installed (pip install matplotlib)
Usage:
    python simple_test.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_segmentation_pipeline import MLTrainingPipeline


def visualize_point_cloud(ply_path, output_image_path, room_name):
    """
    Generate visualization images of the segmented point cloud
    
    Args:
        ply_path: Path to the .ply file
        output_image_path: Where to save the visualization
        room_name: Name of the room for title
    """
    try:
        print("\n📊 Generating visualization images...")
        
        # Read PLY file
        points = []
        colors = []
        
        with open(ply_path, 'r') as f:
            # Skip header
            line = f.readline()
            while 'end_header' not in line:
                line = f.readline()
            
            # Read vertices
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                    points.append([x, y, z])
                    colors.append([r/255.0, g/255.0, b/255.0])
        
        points = np.array(points)
        colors = np.array(colors)
        
        print(f"   Loaded {len(points)} points")
        
        # Downsample for faster rendering (every 10th point)
        step = max(1, len(points) // 50000)  # Keep max 50k points for visualization
        points_ds = points[::step]
        colors_ds = colors[::step]
        
        print(f"   Downsampled to {len(points_ds)} points for visualization")
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'3D Room Segmentation: {room_name}', fontsize=16, fontweight='bold')
        
        # View 1: Top-down view (X-Y plane)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(points_ds[:, 0], points_ds[:, 1], points_ds[:, 2], 
                   c=colors_ds, s=1, alpha=0.6)
        ax1.view_init(elev=90, azim=0)  # Top-down
        ax1.set_title('Top View (Bird\'s Eye)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # View 2: Front view (X-Z plane)
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(points_ds[:, 0], points_ds[:, 1], points_ds[:, 2], 
                   c=colors_ds, s=1, alpha=0.6)
        ax2.view_init(elev=0, azim=0)  # Front
        ax2.set_title('Front View', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # View 3: Side view (Y-Z plane)
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.scatter(points_ds[:, 0], points_ds[:, 1], points_ds[:, 2], 
                   c=colors_ds, s=1, alpha=0.6)
        ax3.view_init(elev=0, azim=90)  # Side
        ax3.set_title('Side View', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # View 4: Isometric view 1
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.scatter(points_ds[:, 0], points_ds[:, 1], points_ds[:, 2], 
                   c=colors_ds, s=1, alpha=0.6)
        ax4.view_init(elev=30, azim=45)
        ax4.set_title('Isometric View 1', fontsize=12, fontweight='bold')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        
        # View 5: Isometric view 2
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.scatter(points_ds[:, 0], points_ds[:, 1], points_ds[:, 2], 
                   c=colors_ds, s=1, alpha=0.6)
        ax5.view_init(elev=30, azim=135)
        ax5.set_title('Isometric View 2', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        
        # View 6: Color legend
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create color legend
        legend_text = """
        COLOR LEGEND
        ═══════════════════════════
        
        🟤 Floor          (Brown)
        🔵 Ceiling        (Light Blue)
        🟠 Wall           (Orange)
        🪟 Window         (Green)
        🚪 Door           (Purple)
        🪑 Chair          (Red-Orange)
        🪑 Table          (Tan/Beige)
        📊 Board          (Light Purple)
        🏛️  Column        (Yellow)
        📦 Clutter        (Gray/White)
        🪜  Sofa          (Pink)
        📚 Bookcase       (Dark Brown)
        ⚠️  Beam          (Dark Gray)
        """
        
        ax6.text(0.1, 0.5, legend_text, fontsize=11, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved: {output_image_path}")
        
        # Close to free memory
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")
        print("   (This is optional - the .ply file was still created successfully)")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  ML ROOM SEGMENTATION TESTING                    ║
║                                                                  ║
║  This script will:                                              ║
║  1. Load the trained model (trained_model.pkl)                 ║
║  2. Process a test room                                        ║
║  3. Predict object labels (chairs, tables, etc.)              ║
║  4. Save segmented result as .ply file                        ║
║  5. Generate visualization images (NEW!)                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    DATASET_PATH = r"E:\3D-semantic\3D-Room-Scene-Semantic-Segmentation\dataset\Stanford3dDataset_v1.2_Aligned_Version"
    MODEL_PATH = "trained_model.pkl"
    OUTPUT_DIR = "test_results"
    
    # Which room to test?
    TEST_AREA = "Area_1"
    TEST_ROOM = "conferenceRoom_1"
    # ========================================================================
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Trained model not found: {MODEL_PATH}")
        print("\nYou need to train the model first!")
        print("Run: python simple_train.py")
        sys.exit(1)
    
    print(f"✓ Model found: {MODEL_PATH}")
    
    # Check if dataset exists
    anno_path = os.path.join(DATASET_PATH, TEST_AREA, TEST_ROOM, "Annotations")
    if not os.path.exists(anno_path):
        print(f"❌ ERROR: Test room not found: {anno_path}")
        print("\nPlease update TEST_AREA and TEST_ROOM in this script.")
        sys.exit(1)
    
    print(f"✓ Test room found: {TEST_AREA}/{TEST_ROOM}\n")
    
    # Initialize pipeline and load model
    print("📥 Loading trained model...")
    pipeline = MLTrainingPipeline(model_path=MODEL_PATH)
    
    if not pipeline.classifier.load_model():
        print("❌ Failed to load model!")
        sys.exit(1)
    
    # Test on room
    print(f"\n🧪 Testing on: {TEST_AREA}/{TEST_ROOM}")
    print("="*70)
    
    success = pipeline.test_on_room(
        anno_path=anno_path,
        room_name=f"{TEST_AREA}_{TEST_ROOM}",
        output_dir=OUTPUT_DIR
    )
    
    if success:
        print("\n" + "="*70)
        print("✅ SUCCESS! Room segmentation complete!")
        print("="*70)
        
        ply_file = os.path.join(OUTPUT_DIR, f"{TEST_AREA}_{TEST_ROOM}_segmented.ply")
        print(f"\n📁 Output saved to: {ply_file}")
        
        # Generate visualization
        viz_file = os.path.join(OUTPUT_DIR, f"{TEST_AREA}_{TEST_ROOM}_visualization.png")
        visualize_point_cloud(ply_file, viz_file, f"{TEST_AREA}/{TEST_ROOM}")
        
        print("\n" + "="*70)
        print("📊 FILES GENERATED:")
        print("="*70)
        print(f"1. 3D Point Cloud: {ply_file}")
        print(f"2. Visualization:  {viz_file}")
        
        print("\n🔍 View the results:")
        print("\nOption 1 - Quick Preview:")
        print(f"  Open: {viz_file}")
        print("  (Shows 6 different angles of the segmented room)")
        
        print("\nOption 2 - Interactive 3D:")
        print(f"  Open: {ply_file}")
        print("  In MeshLab, CloudCompare, or Blender")
        
        print("\n🎨 Colors indicate detected objects:")
        print("  🟤 Brown = Floor")
        print("  🔵 Light Blue = Ceiling")
        print("  🟠 Orange = Walls")
        print("  🔴 Red-Orange = Chairs")
        print("  🟫 Tan = Tables")
        print("  ... and more!")
        
        print("\n💡 For detailed visualization guide, see: docs/meshlab_guide.md")
        
    else:
        print("\n❌ Testing failed. Check error messages above.")


if __name__ == "__main__":
    main()