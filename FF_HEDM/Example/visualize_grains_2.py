import h5py
import numpy as np
import plotly.graph_objects as go

def read_grain_data(filename='OutputFile.hdf'):
    grain_data = {}
    print("\nGrain Information:")
    print("-----------------")
    
    with h5py.File(filename, 'r') as f:
        grains_group = f['Grains']
        for grain_name in ['GrainID_56', 'GrainID_60', 'GrainID_61']:
            # Read SpotMatrix_Radius data for each grain
            spot_matrix = grains_group[f'{grain_name}/SpotMatrix_Radius'][:]
            
            # Get grain volume and radius from first row
            grain_volume = spot_matrix[0, 26]  # GrainVolume column
            grain_radius = spot_matrix[0, 27]  # GrainRadius column
            
            # Calculate grain center (average position of all spots)
            y_centers = spot_matrix[:, 15]  # YCen(px) column
            z_centers = spot_matrix[:, 16]  # ZCen(px) column
            omega = spot_matrix[:, 14]      # Omega(degrees) column
            
            # Print raw Z positions for debugging
            print(f"\n{grain_name}:")
            print(f"  Grain Radius: {grain_radius:.1f}")
            print(f"  Grain Volume: {grain_volume:.1f}")
            
            # Convert omega to x coordinate
            omega_rad = np.radians(omega)
            x_centers = np.cos(omega_rad) * np.mean(y_centers)
            
            grain_center = np.array([
                np.mean(x_centers),
                np.mean(y_centers),
                np.mean(z_centers)
            ])
            
            grain_data[grain_name] = {
                'center': grain_center,
                'volume': grain_volume,
                'radius': grain_radius  # Use actual grain radius
            }
    
    return grain_data

def create_sphere_mesh(center, radius, resolution=30):
    """Create mesh points for a sphere"""
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(-np.pi/2, np.pi/2, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = center[0] + radius * np.cos(theta) * np.cos(phi)
    y = center[1] + radius * np.cos(theta) * np.sin(phi)
    z = center[2] + radius * np.sin(theta)
    
    return x, y, z

def visualize_grains(filename='OutputFile.hdf'):
    # Read data for all grains
    grain_data = read_grain_data(filename)
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for each grain
    colors = {
        'GrainID_56': '#FF1493',  # Deep pink
        'GrainID_60': '#00FF00',  # Bright green
        'GrainID_61': '#4169E1'   # Royal blue
    }
    
    # Scale factor to reduce grain sizes proportionally
    radius_scale = 0.2  # Reduce all radii to 20% of original size
    
    # Calculate axis ranges to ensure equal scaling
    all_centers = np.array([info['center'] for info in grain_data.values()])
    max_radius = max(info['radius'] * radius_scale for info in grain_data.values())
    
    center_mean = np.mean(all_centers, axis=0)
    max_range = max_radius * 3  # Make range 3x the largest radius
    
    # For each grain
    for grain_name, grain_info in grain_data.items():
        center = grain_info['center']
        radius = grain_info['radius'] * radius_scale
        volume = grain_info['volume']
        
        # Create sphere mesh for the grain
        x, y, z = create_sphere_mesh(center, radius)
        
        # Add surface for the grain
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            name=grain_name,
            showscale=False,
            opacity=0.3,
            colorscale=[[0, colors[grain_name]], [1, colors[grain_name]]],
            surfacecolor=np.zeros_like(x),
            hovertemplate=(
                f"<b>{grain_name}</b><br>" +
                "X: %{x:.1f}<br>" +
                "Y: %{y:.1f}<br>" +
                "Z: %{z:.1f}<br>" +
                f"Volume: {volume:.1f}<br>" +
                f"Original Radius: {grain_info['radius']:.1f}<br>" +
                f"Scaled Radius: {radius:.1f}<br>" +
                "<extra></extra>"
            )
        ))
    
    # Update layout with white background and equal axis ranges
    fig.update_layout(
        title="3D Grain Visualization (Real Space)",
        scene=dict(
            xaxis=dict(
                range=[center_mean[0] - max_range, center_mean[0] + max_range],
                title="X"
            ),
            yaxis=dict(
                range=[center_mean[1] - max_range, center_mean[1] + max_range],
                title="Y"
            ),
            zaxis=dict(
                range=[center_mean[2] - max_range, center_mean[2] + max_range],
                title="Z (pixels)"
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=2, y=2, z=2),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        paper_bgcolor='white'
    )
    
    # Save as HTML file for interactive viewing
    fig.write_html("interactive_grains_real_2.html")
    print("\nSaved interactive visualization to interactive_grains_real_2.html")
    
    # Optionally, show in browser directly
    fig.show()

if __name__ == "__main__":
    visualize_grains() 