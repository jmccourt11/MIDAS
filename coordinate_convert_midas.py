#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_peak_data(filename):
    """
    Read peak data from CSV file
    """
    try:
        # First try to read the header to get column names
        with open(filename, 'r') as f:
            header = f.readline().strip()
            if header.startswith('%'):
                header = header[1:]  # Remove % if present
            columns = header.split()
        
        # Read the data with the correct number of columns
        data = pd.read_csv(filename, 
                          delim_whitespace=True,  # Handle variable whitespace
                          skiprows=1,             # Skip header row
                          names=columns,          # Use header names
                          comment='%')            # Skip comment lines
        
        print(f"Successfully read {len(data)} rows with columns: {columns}")
        return data
        
    except Exception as e:
        print(f"Error reading peak data: {str(e)}")
        print(f"File: {filename}")
        print("First few lines of file:")
        try:
            with open(filename, 'r') as f:
                print(f.read(500))  # Print first 500 characters
        except:
            print("Could not read file contents")
        raise

def transform_coordinates(xorig, yorig, omega, detector_distance, xcenter=0, ycenter=0, image_height=None):
    """
    Transform diffraction pattern coordinates to laboratory frame coordinates
    """
    # Correct y coordinates if image_height is provided
    if image_height is not None:
        yorig = image_height - yorig
        ycenter = image_height - ycenter
    
    # Center the coordinates relative to the beam center
    x_centered = xorig - xcenter
    y_centered = yorig - ycenter
    
    # Calculate radius and azimuthal angle in detector plane
    r = np.sqrt(x_centered**2 + y_centered**2)
    eta = np.arctan2(y_centered, x_centered)
    
    # Calculate scattering angle (2θ)
    two_theta = np.arctan2(r, detector_distance)
    
    # Calculate laboratory frame coordinates
    YLab = -detector_distance * np.tan(two_theta) * np.cos(eta)
    ZLab = detector_distance * np.tan(two_theta) * np.sin(eta)
    
    return {
        'YLab': YLab,
        'ZLab': ZLab,
        'eta': eta,
        'two_theta': two_theta,
        'SpotID': np.arange(1, len(xorig) + 1),
        'RingNumber': np.ones_like(xorig)
    }

def reverse_transform(YLab, ZLab, omega, two_theta, detector_distance):
    """
    Perform reverse transformation from laboratory coordinates to detector coordinates
    """
    # Calculate radius in detector plane
    r = np.sqrt(YLab**2 + ZLab**2)
    
    # Calculate eta from YLab and ZLab
    eta = np.arctan2(ZLab, -YLab)  # YLab is negative to right
    
    # Convert back to detector coordinates
    x = r * np.cos(eta)
    y = r * np.sin(eta)
    
    # Shift all coordinates to be positive (move origin to bottom left)
    x_min = np.min(x)
    y_min = np.min(y)
    x = x - x_min
    y = y - y_min
    
    return {'x': x, 'y': y}

def plot_points(coords, xorig, yorig, xcenter=0, ycenter=0, image_height=None, title=None):
    """
    Plot points in both original and transformed coordinate systems
    """
    # For plotting original coordinates, use the imshow convention if image_height is provided
    plot_yorig = yorig if image_height is None else image_height - yorig
    plot_ycenter = ycenter if image_height is None else image_height - ycenter
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # Plot original coordinates
    ax1 = fig.add_subplot(121)
    ax1.scatter(xorig, plot_yorig, c='blue', alpha=0.6, label='Points')
    ax1.scatter(xcenter, plot_ycenter, c='red', marker='x', s=100, label='Beam center')
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlabel('X original')
    ax1.set_ylabel('Y original (imshow convention)')
    if image_height is not None:
        ax1.invert_yaxis()  # To match imshow convention
    ax1.set_title('Original Detector Coordinates')
    ax1.legend()
    
    # Plot transformed coordinates
    ax2 = fig.add_subplot(122)
    ax2.scatter(coords['YLab'], coords['ZLab'], c='red', alpha=0.6)
    ax2.scatter(0, 0, c='red', marker='x', s=100, label='Origin')
    # Add arrows to show coordinate system
    arrow_length = max(abs(coords['YLab'].max()), abs(coords['ZLab'].max())) * 0.2
    ax2.arrow(0, 0, 0, arrow_length, head_width=arrow_length*0.05, 
             head_length=arrow_length*0.1, fc='k', ec='k', label='ZLab axis')
    ax2.arrow(0, 0, -arrow_length, 0, head_width=arrow_length*0.05,
             head_length=arrow_length*0.1, fc='k', ec='k', label='YLab axis')
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlabel('YLab (negative to right)')
    ax2.set_ylabel('ZLab')
    ax2.set_title('Laboratory Frame Coordinates')
    ax2.invert_xaxis()
    ax2.legend()
    
    if title:
        fig.suptitle(title, y=1.05)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)

def plot_all_coordinates(data, detector_distance):
    """
    Plot laboratory and detector coordinates
    """
    detector_coords = reverse_transform(
        data['YLab'], data['ZLab'], 
        data['Omega'], data['Ttheta'], 
        detector_distance
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot laboratory coordinates (YLab negative to right)
    scatter1 = ax1.scatter(data['YLab'], data['ZLab'], 
                          c=data['RingNumber'], cmap='viridis')
    ax1.set_xlabel('YLab (negative to right)')
    ax1.set_ylabel('ZLab')
    ax1.set_title('Laboratory Coordinates')
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.invert_xaxis()  # Ensure YLab is negative to right
    plt.colorbar(scatter1, ax=ax1, label='Ring Number')
    
    # Plot detector coordinates
    scatter2 = ax2.scatter(detector_coords['x'], detector_coords['y'], 
                          c=data['RingNumber'], cmap='viridis')
    ax2.set_xlabel('X Detector')
    ax2.set_ylabel('Y Detector')
    ax2.set_title('Detector Coordinates')
    ax2.grid(True)
    ax2.set_aspect('equal')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    plt.colorbar(scatter2, ax=ax2, label='Ring Number')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def analyze_grain_radius(data):
    """
    Analyze grain radius statistics and create histogram
    """
    stats = {
        'mean': np.mean(data['GrainRadius']),
        'std': np.std(data['GrainRadius']),
        'min': np.min(data['GrainRadius']),
        'max': np.max(data['GrainRadius'])
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data['GrainRadius'], bins=30, edgecolor='black')
    ax.set_xlabel('Grain Radius')
    ax.set_ylabel('Frequency')
    ax.set_title('Grain Radius Distribution')
    ax.grid(True)
    
    return stats, (fig, ax)

def save_transformed_coordinates(data, detector_distance, output_filename, direction='forward'):
    """
    Save transformed coordinates to a file
    """
    # Create copy of data to avoid modifying original
    output_data = data.copy()
    
    if direction == 'forward':
        # Transform laboratory coordinates to detector coordinates
        detector_coords = reverse_transform(
            data['YLab'], data['ZLab'],
            data['Omega'], data['Ttheta'],
            detector_distance
        )
        # Add detector coordinates to output
        output_data['XDetector'] = detector_coords['x']
        output_data['YDetector'] = detector_coords['y']
        
    elif direction == 'reverse':
        # Transform detector coordinates to laboratory coordinates
        lab_coords = transform_coordinates(
            data['XDetector'], data['YDetector'],
            data['Omega'], detector_distance
        )
        # Add laboratory coordinates to output
        output_data['YLab'] = lab_coords['ylab']
        output_data['ZLab'] = lab_coords['zlab']
    
    # Save to file with same format as input
    header = '%YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta'
    if direction == 'forward':
        header += ' XDetector YDetector'
    
    np.savetxt(
        output_filename,
        output_data.to_numpy(),
        header=header,
        comments='',  # Remove the # prefix but keep the % in header
        delimiter=' ',
        fmt='%.5f'
    )
    
    print(f"Saved transformed coordinates to: {output_filename}")

if __name__ == "__main__":
    # Read and analyze data from FF_HEDM/Example/LayerNr_1/InputAll.csv
    filename = "FF_HEDM/Example/LayerNr_1/InputAll.csv"
    detector_distance = 10000
    
    # Read data
    data = read_peak_data(filename)
    
    # Transform coordinates
    coords = transform_coordinates(
        xorig=data['XDetector'], 
        yorig=data['YDetector'], 
        omega=np.radians(data['Omega']),
        detector_distance=detector_distance,
        xcenter=640,  # beam center x
        ycenter=640   # beam center y
    )
    
    # Plot the transformed coordinates
    plot_points(
        coords=coords,
        xorig=data['XDetector'],
        yorig=data['YDetector'],
        xcenter=640,
        ycenter=640,
        title="FF_HEDM Example Data Transformation"
    )
    
    # Save transformed coordinates
    save_transformed_coordinates(data, detector_distance, "FF_HEDM_transformed.csv", direction='forward')
    
    # Analyze grain radius
    stats, (fig_grain, axes_grain) = analyze_grain_radius(data)
    print("\nGrain Radius Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Plot coordinates
    fig_coords, axes_coords = plot_all_coordinates(data, detector_distance)
    
    # Print summary of the data
    print("\nData Summary:")
    print(data.describe())
    
    plt.show()
# %%
