import streamlit as st
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import os
import tempfile
import glob
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="3D Point Cloud Visualizer",
    page_icon="🧊",
    layout="wide"
)

def load_point_cloud(file_input, file_path=None):
    """
    Load point cloud from PLY or TXT file.
    Returns points (Nx3) and colors (Nx3).
    """
    try:
        pcd = o3d.geometry.PointCloud()
        
        # Determine file type and path
        filename = ""
        if isinstance(file_input, str):
            filename = file_input
            path = file_input
        else:
            filename = file_input.name
            # Save uploaded buffer to temp file
            suffix = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_input.getvalue())
                path = tmp_file.name

        if filename.lower().endswith('.ply'):
            pcd = o3d.io.read_point_cloud(path)
            
            if not isinstance(file_input, str):
                os.remove(path) # Clean up temp file

            if not pcd.has_points():
                return None, None

            points = np.asarray(pcd.points)
            colors = None
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                
        elif filename.lower().endswith('.txt'):
            # Assume XYZRGB or XYZ format
            try:
                # Use pandas for faster loading, handle multiple spaces
                df = pd.read_csv(path, sep=r'\s+', header=None, comment='#', engine='python')
                data = df.values
                
                # Cleanup temp file if needed
                if not isinstance(file_input, str):
                    os.remove(path)
                
                if data.shape[1] >= 3:
                    points = data[:, :3]
                else:
                    return None, None
                    
                if data.shape[1] >= 6:
                    colors = data[:, 3:6]
                    # Normalize if 0-255
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                else:
                    colors = None
            except Exception as e:
                if not isinstance(file_input, str):
                    os.remove(path)
                raise e
        else:
             return None, None
        
        return points, colors

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

def visualize_point_cloud(points, colors, point_size=2):
    """
    Visualize point cloud using Plotly
    """
    if points is None:
        return

    # Subsample if too large to prevent browser crash
    max_points = 50000
    if len(points) > max_points:
        st.warning(f"Point cloud has {len(points):,} points. Downsampling to {max_points:,} for better performance.")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    # Prepare colors for Plotly
    if colors is not None:
        # If float 0-1, convert to int 0-255 for string formatting if needed, 
        # but plotly accepts arrays of floats 0-1 or rgb string. 
        # Let's simply pass the array, Plotly handles it.
        pass
    else:
        colors = 'blue' # Default color

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=colors,
            opacity=0.8,
            colorscale='Viridis' if isinstance(colors, np.ndarray) and colors.ndim == 1 else None 
            # Note: if colors is Nx3 RGB, plotly might interprete differently depending on version.
            # Safe way for RGB is to convert to hex strings or use 'color' arg with list of rgb strings.
        )
    )])

    # If colors is Nx3 RGB array, we need to convert to list of CSS colors for Plotly
    if isinstance(colors, np.ndarray) and colors.ndim == 2 and colors.shape[1] == 3:
        # Assuming Open3D 0-1 float range
        cols = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r,g,b in colors]
        fig.update_traces(marker=dict(color=cols))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(r=0, l=0, b=0, t=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Application UI
st.title("3D Room Scene Visualizer")
st.markdown("Upload a PLY/TXT file or choose from the dataset to visualize.")

# Sidebar controls
st.sidebar.header("Configuration")
source_option = st.sidebar.radio("Data Source", ["Upload File", "Dataset Browser"])
point_size = st.sidebar.slider("Point Size", 1, 10, 2)

selected_file = None
points = None
colors = None

if source_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload a file (.ply, .txt)", type=['ply', 'txt'])
    if uploaded_file is not None:
        with st.spinner("Loading point cloud..."):
            points, colors = load_point_cloud(uploaded_file)
            st.sidebar.success(f"Loaded {len(points):,} points")

elif source_option == "Dataset Browser":
        # Define base paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    paths_to_scan = {
        "Test Results": os.path.join(project_root, "codes", "test_results"),
        "Dataset": os.path.join(project_root, "datatset")
    }    
    selected_category = st.sidebar.selectbox("Category", list(paths_to_scan.keys()))
    base_path = paths_to_scan[selected_category]
    
    if os.path.exists(base_path): 
        # Recursive glob to find files
        search_patterns = ["**/*.ply", "**/*.txt"]
        files = []
        for pattern in search_patterns:
            # Use glob to find files, recursive=True requires valid pattern
            found = glob.glob(os.path.join(base_path, pattern), recursive=True)
            files.extend(found)
            
            
        if files:
            # Sort files for better UX
            files.sort()
            
            # Make paths relative for display
            rel_files = [os.path.relpath(f, base_path) for f in files]
            
            st.sidebar.markdown(f"**Found {len(files)} files**")
            selected_rel_file = st.sidebar.selectbox("Select File", rel_files)
            selected_path = os.path.join(base_path, selected_rel_file)
            
            if st.sidebar.button("Load Selected File"):
                with st.spinner(f"Loading {selected_rel_file}..."):
                    points, colors = load_point_cloud(selected_path)
                    if points is not None:
                        st.sidebar.success(f"Loaded {len(points):,} points")
        else:
            st.sidebar.warning(f"No compatible files found in {selected_category}")
    else:
        st.sidebar.error(f"Directory not found: {base_path}")

# Visualization logic
if points is not None:
    st.subheader("3D Visualization")
    visualize_point_cloud(points, colors, point_size)
    
    # Check if we should explain the controls
    with st.expander("Usage Guide"):
        st.markdown("""
        - **Rotate**: Left-click and drag
        - **Pan**: Right-click and drag
        - **Zoom**: Scroll
        - **Reset Camera**: Double-click
        """)
else:
    if source_option == "Upload File" and not uploaded_file:
        st.info("Please upload a .ply or .txt file to begin.")
    elif source_option == "Upload File" and uploaded_file:
         # Failed to load (error message already shown in load_ply)
         pass

