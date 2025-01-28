#%%
"""
coordinate_convert_midas_cindy.py

This module extends coordinate_convert_midas.py to handle peak analysis data from CSV files.
It provides functions to read, transform, and visualize peak positions and intensities.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from coordinate_convert_midas import transform_coordinates, plot_points
from tqdm import tqdm

def analyze_peak_data(filename, detector_distance=10000, omega=0, xcenter=0, ycenter=0):
    """
    Read and analyze peak data from CSV file, focusing on the frame with maximum intensity
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file containing peak data
    detector_distance : float
        Distance from sample to detector in mm
    omega : float
        Sample rotation angle in degrees
    xcenter : float
        Beam center X coordinate
    ycenter : float
        Beam center Y coordinate
        
    Returns:
    --------
    dict containing:
        coords : dict
            Transformed coordinates from transform_coordinates()
        figures : tuple
            (fig, (ax1, ax2, ax3)) containing the plot objects
        peak_data : pandas DataFrame
            Peak data for the frame with maximum intensity
    """
    # Read the CSV file
    peak_data = pd.read_csv(filename)

    # Print the structure of the data
    print("Data columns:")
    print(peak_data.columns.tolist())
    
    # Find the frame with maximum total intensity
    frame_intensities = peak_data.groupby('frame')['intensity'].sum()
    max_intensity_frame = frame_intensities.idxmax()
    
    print(f"\nAnalyzing frame {max_intensity_frame} (highest total intensity)")
    
    # Filter data for the maximum intensity frame
    frame_data = peak_data[peak_data['frame'] == max_intensity_frame].copy()
    
    print("\nFirst few rows of selected frame:")
    print(frame_data.head())

    # Extract coordinates and intensities
    x_coords = frame_data['x']
    y_coords = frame_data['y']
    intensities = frame_data['intensity']

    # Transform coordinates
    coords = transform_coordinates(
        xorig=x_coords,
        yorig=y_coords,
        omega=np.radians(omega),
        detector_distance=detector_distance,
        xcenter=xcenter,
        ycenter=ycenter
    )

    # Create visualization
    fig = plt.figure(figsize=(15, 5))

    # Original coordinates colored by intensity
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(x_coords, y_coords, c=intensities, cmap='viridis')
    ax1.set_title(f'Frame {max_intensity_frame}\nOriginal Coordinates')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(scatter1, ax=ax1, label='Intensity')

    # Transformed coordinates (YLab vs ZLab)
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(coords['YLab'], coords['ZLab'], c=intensities, cmap='viridis')
    ax2.set_title('Lab Frame Coordinates')
    ax2.set_xlabel('YLab')
    ax2.set_ylabel('ZLab')
    ax2.invert_xaxis()  # YLab is negative to right
    plt.colorbar(scatter2, ax=ax2, label='Intensity')

    # 2θ vs η plot
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(np.degrees(coords['two_theta']), 
                          np.degrees(coords['eta']), 
                          c=intensities, 
                          cmap='viridis')
    ax3.set_title('2θ vs η')
    ax3.set_xlabel('2θ (degrees)')
    ax3.set_ylabel('η (degrees)')
    plt.colorbar(scatter3, ax=ax3, label='Intensity')

    plt.tight_layout()

    # Print summary statistics
    print("\nSummary Statistics for frame {}:".format(max_intensity_frame))
    print(f"Number of peaks: {len(x_coords)}")
    print(f"2θ range: {np.degrees(coords['two_theta'].min()):.2f}° to {np.degrees(coords['two_theta'].max()):.2f}°")
    print(f"η range: {np.degrees(coords['eta'].min()):.2f}° to {np.degrees(coords['eta'].max()):.2f}°")
    print(f"Intensity range: {intensities.min():.2f} to {intensities.max():.2f}")
    print(f"Total intensity: {intensities.sum():.2f}")

    return {
        'coords': coords,
        'figures': (fig, (ax1, ax2, ax3)),
        'peak_data': frame_data,
        'frame_number': max_intensity_frame
    }

def create_frame_grid(filename, grid_size_col=20, grid_size_row=15, image_size=(250, 250)):
    """
    Create a grid visualization of all frames from a CSV file
    
    Parameters:
    -----------
    filename : str
        Path to CSV file containing peak data
    grid_size_col : int
        Number of columns in the grid
    grid_size_row : int
        Number of rows in the grid
    image_size : tuple
        Size of each frame (height, width)
    """
    # Read data
    data = pd.read_csv(filename)
    
    # Create figure
    total_width = grid_size_col * image_size[1]
    total_height = grid_size_row * image_size[0]
    #fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    fig, ax = plt.subplots(1, 1, figsize=(11, 12))
    
    # Get unique frames
    frames = sorted(data['frame'].unique())
    
    # Normalize intensities for alpha scaling
    max_intensity = data['intensity'].max()
    min_intensity = data['intensity'].min()
    
    # Plot each frame in the grid
    for i in tqdm(range(grid_size_row), desc="Creating grid visualization"):
        for j in range(grid_size_col):
            idx = i * grid_size_col + j
            if idx < len(frames):
                # Get frame data
                frame_data = data[data['frame'] == frames[idx]]
                
                # Calculate offsets
                x_offset = j * image_size[1]
                y_offset = i * image_size[0]
                
                # Normalize intensities to [0.1, 1.0] range for alpha
                alphas = 0.1 + 0.9 * (frame_data['intensity'] - min_intensity) / (max_intensity - min_intensity)
                
                # Plot peaks
                ax.scatter(
                    frame_data['y'] + x_offset,
                    frame_data['x'] + y_offset,
                    c='white',  # Use white color for all peaks
                    s=10,  # Larger point size
                    alpha=alphas,  # Use normalized intensities for alpha
                )
    
    # Configure plot
    ax.set_facecolor('black')  # Black background
    fig.patch.set_facecolor('black')  # Black background for the figure
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis('off')
    
    # Adjust layout
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Display plot
    plt.show()
#%%
if __name__ == "__main__":
    # Example usage
    scan_number = 1635
    filename = f'/net/micdata/data2/12IDC/ptychosaxs/peak_analysis/peak_analysis_scan_{scan_number}.csv'
    
    results = analyze_peak_data(
        filename,
        detector_distance=10000,  # Adjust as needed
        omega=0,                 # Adjust as needed
        xcenter=256,              # Adjust as needed
        ycenter=256               # Adjust as needed
    )
    
    plt.show()

    filename = f'/net/micdata/data2/12IDC/ptychosaxs/peak_analysis/peak_analysis_scan_{scan_number}.csv'
    create_frame_grid(filename,grid_size_col=11, grid_size_row=12,image_size=(512, 512))
# %%
