import os
import numpy as np
from pathlib import Path

def explore_dataset_structure(base_path):
    """
    Explore the uploaded S3DIS dataset structure
    """
    print("=" * 60)
    print("DATASET STRUCTURE EXPLORATION")
    print("=" * 60)
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"❌ Path not found: {base_path}")
        return None
    
    print(f"✓ Base path found: {base_path}\n")
    
    # List all areas
    areas = sorted([d for d in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, d)) and d.startswith('Area')])
    
    print(f"Found {len(areas)} areas:")
    for area in areas:
        print(f"  - {area}")
    
    # Explore each area
    dataset_info = {}
    for area in areas:
        area_path = os.path.join(base_path, area)
        rooms = sorted([d for d in os.listdir(area_path) 
                       if os.path.isdir(os.path.join(area_path, d))])
        
        print(f"\n{area}: {len(rooms)} rooms")
        dataset_info[area] = []
        
        for room in rooms[:3]:  # Show first 3 rooms
            room_path = os.path.join(area_path, room)
            print(f"    - {room}")
            
            # Check for Annotations folder
            anno_path = os.path.join(room_path, 'Annotations')
            if os.path.exists(anno_path):
                objects = [f for f in os.listdir(anno_path) if f.endswith('.txt')]
                print(f"        Objects: {len(objects)} files")
                dataset_info[area].append({
                    'room': room,
                    'path': room_path,
                    'anno_path': anno_path,
                    'num_objects': len(objects)
                })
        
        if len(rooms) > 3:
            print(f"    ... and {len(rooms) - 3} more rooms")
    
    return dataset_info


def load_room_from_annotations(anno_path):
    """
    Load point cloud data from S3DIS Annotations folder.
    Each object is stored as a separate .txt file.
    
    Format: X Y Z R G B
    """
    print(f"\n{'=' * 60}")
    print("LOADING ROOM DATA FROM ANNOTATIONS")
    print(f"{'=' * 60}")
    print(f"Path: {anno_path}\n")
    
    if not os.path.exists(anno_path):
        print(f"❌ Annotations path not found: {anno_path}")
        return None, None
    
    # Get all object files
    object_files = sorted([f for f in os.listdir(anno_path) if f.endswith('.txt')])
    print(f"Found {len(object_files)} object files:")
    
    all_points = []
    all_colors = []
    all_labels = []
    
    for idx, obj_file in enumerate(object_files):
        obj_path = os.path.join(anno_path, obj_file)
        object_name = obj_file.replace('.txt', '')
        
        try:
            # Load the file
            data = np.loadtxt(obj_path)
            
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            num_points = data.shape[0]
            
            # Extract coordinates and colors
            points = data[:, :3]  # X, Y, Z
            colors = data[:, 3:6] if data.shape[1] >= 6 else np.ones((num_points, 3)) * 128
            
            # Store data
            all_points.append(points)
            all_colors.append(colors)
            all_labels.append(np.full(num_points, idx))  # Label by object index
            
            print(f"  {idx:2d}. {object_name:30s} - {num_points:8d} points")
            
        except Exception as e:
            print(f"  ⚠️  Error loading {obj_file}: {str(e)}")
    
    # Combine all objects into single point cloud
    if all_points:
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        labels = np.concatenate(all_labels)
        
        print(f"\n{'─' * 60}")
        print(f"TOTAL ROOM DATA:")
        print(f"  Total Points: {len(points):,}")
        print(f"  Total Objects: {len(object_files)}")
        print(f"  Point Cloud Bounds:")
        print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(f"{'─' * 60}")
        
        return {
            'points': points,
            'colors': colors,
            'labels': labels,
            'object_names': [f.replace('.txt', '') for f in object_files]
        }, object_files
    
    return None, None


# Main execution
if __name__ == "__main__":
    # Define your dataset base path
    # MODIFY THIS PATH based on where you uploaded the data
    BASE_PATH = r"E:\3D-semantic\3D-Room-Scene-Semantic-Segmentation\dataset\Stanford3dDataset_v1.2_Aligned_Version"
    
    # Step 1: Explore dataset structure
    dataset_info = explore_dataset_structure(BASE_PATH)
    
    if dataset_info and len(dataset_info) > 0:
        # Step 2: Load a sample room (Area_1, first office)
        first_area = list(dataset_info.keys())[0]
        if dataset_info[first_area]:
            sample_room = dataset_info[first_area][0]
            print(f"\n\n{'=' * 60}")
            print(f"LOADING SAMPLE ROOM: {sample_room['room']}")
            print(f"{'=' * 60}")
            
            room_data, object_files = load_room_from_annotations(sample_room['anno_path'])
            
            if room_data:
                print("\n✓ Successfully loaded room data!")
                print(f"\nRoom ready for segmentation with {len(room_data['points']):,} points")
    else:
        print("\n❌ Could not explore dataset. Please check the base path.")