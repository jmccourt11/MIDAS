import h5py
import numpy as np
import plotly.graph_objects as go

def read_grain_data(filename='OutputFile.hdf'):
    grain_data = {}
    with h5py.File(filename, 'r') as f:
        grains_group = f['Grains']
        for grain_name in ['GrainID_56', 'GrainID_60', 'GrainID_61']:
            # Read SpotMatrix_Radius data for each grain
            spot_matrix = grains_group[f'{grain_name}/SpotMatrix_Radius'][:]
            grain_data[grain_name] = spot_matrix
    return grain_data

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
    
    # For each grain
    for grain_name, spot_matrix in grain_data.items():
        # Extract relevant columns
        y_centers = spot_matrix[:, 3]  # YCen column
        z_centers = spot_matrix[:, 4]  # ZCen column
        omega = spot_matrix[:, 2]      # Omega column
        radii = np.abs(spot_matrix[:, 8])  # Radius column, using absolute values
        intensities = spot_matrix[:, 1]  # Integrated Intensity
        
        # Scale radii to reasonable size for visualization
        scaled_radii = 5 + (radii - np.min(radii)) / (np.max(radii) - np.min(radii)) * 15
        
        # Convert omega to x coordinate
        omega_rad = np.radians(omega)
        x_centers = np.cos(omega_rad) * 100
        
        # Add filled markers
        fig.add_trace(go.Scatter3d(
            x=x_centers,
            y=y_centers,
            z=z_centers,
            mode='markers',
            name=grain_name,
            marker=dict(
                size=scaled_radii,
                color=intensities,
                colorscale='Viridis',
                opacity=0.7,
                line=dict(
                    color=colors[grain_name],
                    width=2
                ),
                colorbar=dict(
                    title="Intensity",
                    len=0.8,
                    y=0.8
                )
            ),
            hovertemplate=(
                f"<b>{grain_name}</b><br>" +
                "X: %{x:.1f}<br>" +
                "Y: %{y:.1f}<br>" +
                "Z: %{z:.1f}<br>" +
                "Intensity: %{marker.color:.1f}<br>" +
                "Original Radius: %{text:.1f}<br>" +
                "<extra></extra>"
            ),
            text=radii,
            showlegend=True
        ))
    
    # Update layout with white background
    fig.update_layout(
        title="3D Diffraction Spot Visualization",
        scene=dict(
            xaxis_title="X (Omega projection)",
            yaxis_title="Y (pixels)",
            zaxis_title="Z (pixels)",
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
            x=0.99,
            itemsizing='constant'
        ),
        paper_bgcolor='white'
    )
    
    # Save as HTML file for interactive viewing
    fig.write_html("interactive_grains_spots.html")
    print("Saved interactive visualization to interactive_grains_spots.html")
    
    # Optionally, show in browser directly
    fig.show()

if __name__ == "__main__":
    visualize_grains() 