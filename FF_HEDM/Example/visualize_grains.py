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
            grain_volume = spot_matrix[0, 14]  # GrainVolume column
            grain_radius = spot_matrix[0, 15]  # GrainRadius column
            
            # Calculate grain center (average position of all spots)
            y_centers = spot_matrix[:, 3]  # YCen column
            z_centers = spot_matrix[:, 4]  # ZCen column
            omega = spot_matrix[:, 2]      # Omega column
            
            # Print raw Z positions for debugging
            print(f"\n{grain_name} Z positions:")
            print(f"  Min Z: {np.min(z_centers):.1f}")
            print(f"  Max Z: {np.max(z_centers):.1f}")
            print(f"  Mean Z: {np.mean(z_centers):.1f}")
            
            # Convert omega to x coordinate using both sin and cos for better position estimate
            omega_rad = np.radians(omega)
            x_centers = np.cos(omega_rad) * np.mean(y_centers)
            
            grain_center = np.array([
                np.mean(x_centers),
                np.mean(y_centers),
                np.mean(z_centers)
            ])
            
            # Scale radius to be visible (original might be too small or large)
            scaled_radius = np.sqrt(abs(grain_volume)) / 10  # Using volume to determine size
            
            grain_data[grain_name] = {
                'center': grain_center,
                'volume': grain_volume,
                'radius': scaled_radius
            }
            
            print(f"\n{grain_name}:")
            print(f"  Center: ({grain_center[0]:.1f}, {grain_center[1]:.1f}, {grain_center[2]:.1f})")
            print(f"  Original Volume: {grain_volume:.1f}")
            print(f"  Original Radius: {grain_radius:.1f}")
            print(f"  Scaled Radius: {scaled_radius:.1f}")
    
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
    
    # Looking at the Z positions, all grains are around Z=1000
    # Let's center them by subtracting the mean Z
    z_offset = 1000
    
    # For each grain
    for grain_name, grain_info in grain_data.items():
        center = grain_info['center']
        # Center the Z coordinate
        center[2] = center[2] - z_offset
        radius = grain_info['radius']
        volume = grain_info['volume']
        
        # Create sphere mesh for the grain
        x, y, z = create_sphere_mesh(center, radius)
        
        # Add surface for the grain
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            name=grain_name,
            showscale=False,
            opacity=0.6,
            colorscale=[[0, colors[grain_name]], [1, colors[grain_name]]],
            surfacecolor=np.zeros_like(x),
            hovertemplate=(
                f"<b>{grain_name}</b><br>" +
                "X: %{x:.1f}<br>" +
                "Y: %{y:.1f}<br>" +
                "Z: %{z:.1f}<br>" +
                f"Volume: {volume:.1f}<br>" +
                f"Radius: {radius:.1f}<br>" +
                "<extra></extra>"
            )
        ))
    
    # Update layout with white background
    fig.update_layout(
        title="3D Grain Visualization (Real Space)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (pixels - 1000)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
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
    fig.write_html("interactive_grains_real.html")
    print("\nSaved interactive visualization to interactive_grains_real.html")
    
    # Optionally, show in browser directly
    fig.show()

if __name__ == "__main__":
    visualize_grains() 