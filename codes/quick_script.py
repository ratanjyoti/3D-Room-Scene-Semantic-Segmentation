"""
QUICK START SCRIPT - 3D Room Segmentation
Run this script first to test your dataset and see results!

This script will:
1. Explore your uploaded dataset structure
2. Process one sample room (Area_1/office_1 by default)
3. Generate visualizations and segmented output
"""

import os
import sys
import numpy as np
from pathlib import Path

# ==============================================================================
# STEP 0: CHECK DATASET
# ==============================================================================

def find_dataset_path():
    """Automatically find the S3DIS dataset in current directory."""
    
    possible_names = [
        "Stanford3dDataset_v1.2_Aligned_Version",
        "Stanford3dDataset_v1.2",
        "S3DIS",
        "dataset",
        "Area_1"
    ]
    
    print("Searching for S3DIS dataset...")
    
    # Check current directory
    for name in possible_names:
        if os.path.exists(name):
            print(f"✓ Found dataset: {name}")
            return name
    
    # Check if Area_1 is directly accessible
    if os.path.exists("Area_1"):
        print("✓ Found Area_1 directory directly")
        return "."
    
    print("❌ Dataset not found in current directory")
    print("\nPlease specify the dataset path manually:")
    print("  1. Update DATASET_PATH variable in this script")
    print("  2. Or run: python script.py /path/to/dataset")
    return None


def explore_dataset(base_path):
    """Explore dataset structure and find available rooms."""
    
    print(f"\n{'='*70}")
    print("DATASET EXPLORATION")
    print(f"{'='*70}\n")
    
    if not os.path.exists(base_path):
        print(f"❌ Path not found: {base_path}")
        return None
    
    # Find all areas
    areas = sorted([d for d in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, d)) 
                   and d.startswith('Area')])
    
    if not areas:
        print("❌ No Area folders found!")
        return None
    
    print(f"Found {len(areas)} areas: {', '.join(areas)}\n")
    
    available_rooms = []
    
    for area in areas:
        area_path = os.path.join(base_path, area)
        rooms = sorted([d for d in os.listdir(area_path) 
                       if os.path.isdir(os.path.join(area_path, d))])
        
        print(f"{area}: {len(rooms)} rooms")
        
        for room in rooms[:3]:  # Show first 3
            room_path = os.path.join(area_path, room)
            anno_path = os.path.join(room_path, 'Annotations')
            
            if os.path.exists(anno_path):
                num_files = len([f for f in os.listdir(anno_path) if f.endswith('.txt')])
                print(f"  ✓ {room} ({num_files} objects)")
                available_rooms.append({
                    'area': area,
                    'room': room,
                    'path': anno_path,
                    'num_objects': num_files
                })
            else:
                print(f"  ✗ {room} (no Annotations folder)")
        
        if len(rooms) > 3:
            print(f"  ... and {len(rooms)-3} more rooms")
        print()
    
    return available_rooms


# ==============================================================================
# STEP 1: SIMPLIFIED PIPELINE FOR QUICK TESTING
# ==============================================================================

def quick_segment_room(anno_path, output_name="test_room"):
    """
    Simplified segmentation for quick testing.
    Uses all the modules but with minimal configuration.
    """
    
    print(f"\n{'='*70}")
    print(f"QUICK SEGMENTATION: {output_name}")
    print(f"{'='*70}\n")
    
    # 1. Load data
    print("📂 Loading point cloud data...")
    object_files = sorted([f for f in os.listdir(anno_path) if f.endswith('.txt')])
    
    all_points = []
    all_colors = []
    
    for obj_file in object_files:
        data = np.loadtxt(os.path.join(anno_path, obj_file))
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        all_points.append(data[:, :3])
        all_colors.append(data[:, 3:6] if data.shape[1] >= 6 else np.ones((data.shape[0], 3)) * 128)
    
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    print(f"  Loaded {len(points):,} points from {len(object_files)} objects")
    
    # 2. Quick preprocessing (just voxel downsample)
    print("\n🔧 Preprocessing...")
    voxel_size = 0.03
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    unique_voxels = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in unique_voxels:
            unique_voxels[key] = {'pts': [points[i]], 'cols': [colors[i]]}
        else:
            unique_voxels[key]['pts'].append(points[i])
            unique_voxels[key]['cols'].append(colors[i])
    
    ds_points = np.array([np.mean(v['pts'], axis=0) for v in unique_voxels.values()])
    ds_colors = np.array([np.mean(v['cols'], axis=0) for v in unique_voxels.values()])
    
    print(f"  Downsampled to {len(ds_points):,} points")
    
    # 3. Quick clustering
    print("\n🔍 Clustering...")
    from sklearn.cluster import DBSCAN
    
    clustering = DBSCAN(eps=0.05, min_samples=10, n_jobs=-1)
    labels = clustering.fit_predict(ds_points)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"  Found {num_clusters} clusters")
    
    # 4. Simple labeling (just by height)
    print("\n🏷️  Labeling...")
    z_min, z_max = ds_points[:, 2].min(), ds_points[:, 2].max()
    z_range = z_max - z_min
    
    semantic_labels = np.zeros(len(ds_points), dtype=int)
    
    # Floor: lowest 15%
    floor_mask = ds_points[:, 2] < z_min + 0.15 * z_range
    semantic_labels[floor_mask] = 1
    
    # Ceiling: highest 15%
    ceiling_mask = ds_points[:, 2] > z_min + 0.85 * z_range
    semantic_labels[ceiling_mask] = 2
    
    # Walls and furniture (rough classification)
    remaining = ~floor_mask & ~ceiling_mask
    semantic_labels[remaining] = 3  # Default to walls/furniture
    
    # Color mapping
    label_colors = {
        0: [128, 128, 128],  # Unknown - gray
        1: [139, 69, 19],    # Floor - brown
        2: [173, 216, 230],  # Ceiling - light blue
        3: [144, 238, 144]   # Walls/Furniture - light green
    }
    
    seg_colors = np.array([label_colors[l] for l in semantic_labels])
    
    print(f"  Floor: {np.sum(semantic_labels==1):,} points")
    print(f"  Ceiling: {np.sum(semantic_labels==2):,} points")
    print(f"  Walls/Furniture: {np.sum(semantic_labels==3):,} points")
    
    # 5. Save simple PLY output
    print("\n💾 Exporting PLY...")
    output_file = f"{output_name}_quick_segmented.ply"
    
    with open(output_file, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(ds_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(ds_points)):
            f.write(f"{ds_points[i,0]:.6f} {ds_points[i,1]:.6f} {ds_points[i,2]:.6f} ")
            f.write(f"{int(seg_colors[i,0])} {int(seg_colors[i,1])} {int(seg_colors[i,2])}\n")
    
    print(f"  ✓ Saved: {output_file}")
    
    # 6. Create simple visualization
    print("\n📊 Creating visualization...")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 7))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    sample_idx = np.random.choice(len(ds_points), min(10000, len(ds_points)), replace=False)
    ax1.scatter(ds_points[sample_idx, 0], ds_points[sample_idx, 1], ds_points[sample_idx, 2],
               c=ds_colors[sample_idx]/255, s=1, alpha=0.6)
    ax1.set_title('Original')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Segmented
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(ds_points[sample_idx, 0], ds_points[sample_idx, 1], ds_points[sample_idx, 2],
               c=seg_colors[sample_idx]/255, s=1, alpha=0.6)
    ax2.set_title('Segmented')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    viz_file = f"{output_name}_quick_visualization.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_file}")
    plt.close()
    
    print(f"\n{'='*70}")
    print("✅ QUICK SEGMENTATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {viz_file}")
    print(f"\nYou can view the .ply file in MeshLab, CloudCompare, or any PLY viewer.")
    

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print(" 3D ROOM SEGMENTATION - QUICK START")
    print("="*70 + "\n")
    
    # Find dataset
    DATASET_PATH = r"E:\3D-semantic\3D-Room-Scene-Semantic-Segmentation\dataset\Stanford3dDataset_v1.2_Aligned_Version"

    if DATASET_PATH is None:
        print("\n⚠️  Please specify dataset path and try again.")
        sys.exit(1)
    
    # Explore dataset
    available_rooms = explore_dataset(DATASET_PATH)
    
    if not available_rooms:
        print("❌ No valid rooms found!")
        sys.exit(1)
    
    # Process first available room
    first_room = available_rooms[0]
    
    print(f"\n{'='*70}")
    print(f"Processing: {first_room['area']}/{first_room['room']}")
    print(f"{'='*70}")
    
    quick_segment_room(
        first_room['path'],
        output_name=f"{first_room['area']}_{first_room['room']}"
    )
    
    print("\n✅ All done! Check the output files above.")
    print("\n💡 Next steps:")
    print("   1. View the .ply file in a 3D viewer (MeshLab, CloudCompare)")
    print("   2. Check the visualization PNG")
    print("   3. Run the full pipeline for better results (see main_pipeline.py)")