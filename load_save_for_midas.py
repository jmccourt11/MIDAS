#!/usr/bin/env python3
"""
load_save_for_midas.py

Script to load data from various sources and compile into a single HDF5 file for MIDAS.
Handles specific data structures from ptychography reconstruction files.
"""

import h5py
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Union, List, Any
from scipy import io as sio
import pandas as pd
from coordinate_convert_midas_cindy import analyze_peak_data

def update_paths_with_scan(config: Dict[str, str], scan_number: int) -> Dict[str, str]:
    """
    Update file paths in config with the specified scan number,
    keeping 'tomo_scan3' constant and preserving cell_info path
    
    Parameters
    ----------
    config : dict
        Original config dictionary
    scan_number : int
        Scan number to use in paths
    
    Returns
    -------
    dict
        Updated config dictionary
    """
    updated_config = config.copy()
    
    # Replace scan number in each path
    for key in updated_config:
        # Skip cell_info path completely
        if key == 'cell_info':
            print(f"Preserving cell_info path: {updated_config[key]}")
            continue
            
        path = updated_config[key]
        import re
        
        if 'ptycho' in path:
            # Handle master file path differently
            updated_path = re.sub(r'/scan\d+/scan\d+_master', 
                                f'/scan{scan_number}/scan{scan_number}_master', 
                                path)
        else:
            # Handle other paths (keeping tomo_scan3 unchanged)
            updated_path = re.sub(r'/scan\d+/', f'/scan{scan_number}/', path)
        
        updated_config[key] = updated_path
        
        print(f"Original path: {path}")
        print(f"Updated path: {updated_path}")
    
    return updated_config

def extract_cell_info(cell_file: str) -> Dict[str, np.ndarray]:
    """
    Extract unit cell information from MATLAB file
    
    Parameters
    ----------
    cell_file : str
        Path to cellinfo.mat file
        
    Returns
    -------
    dict
        Dictionary containing unit cell parameters
    """
    print("\nLoading cell information...")
    try:
        # Print the exact path being used
        print(f"Attempting to read: {cell_file}")
        
        # Load the .mat file
        cell_data = sio.loadmat(cell_file)
        
        # Extract the structured array
        cellinfo_struct = cell_data['cellinfo'][0, 0]
        
        # Define which fields to extract
        wanted_fields = ['A', 'B', 'C', 'alpha', 'beta', 'gamma']
        
        # Create dictionary with only the wanted fields
        cell_info = {}
        for name in wanted_fields:
            if name in cellinfo_struct.dtype.names:
                # Extract the value and remove unnecessary nesting
                value = cellinfo_struct[name][0, 0]
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = value.item()  # Convert single-element arrays to scalars
                cell_info[name] = value
                print(f"  {name}: {value}")
        
        print("Successfully loaded cell information")
        return cell_info
        
    except Exception as e:
        print(f"Error reading cell information: {str(e)}")
        print(f"File path: {cell_file}")
        raise  # Raise the error to see the full traceback

def extract_diffraction_data(master_file: str) -> Dict[str, np.ndarray]:
    """
    Extract diffraction patterns from master HDF5 file
    
    Parameters
    ----------
    master_file : str
        Path to master HDF5 file containing diffraction data
        
    Returns
    -------
    dict
        Dictionary containing:
        - diffraction_patterns: 3D array of all patterns (N_patterns x height x width)
        - row_indices: Dictionary mapping row number to slice indices
    """
    diffraction_patterns = []
    row_indices = {}
    start_idx = 0
    
    print("\nLoading diffraction patterns from rows:")
    try:
        with h5py.File(master_file, 'r') as f:
            # Get data group
            data_group = f['entry']['data']
            
            # Loop through each row's data
            patterns_found = False
            for i in range(1, 13):  # 12 rows
                key = f'data_{i:06d}'  # Zero-pad to 6 digits
                if key in data_group:
                    patterns = data_group[key][()]
                    n_patterns = patterns.shape[0]
                    
                    # Store indices for this row
                    row_indices[i] = slice(start_idx, start_idx + n_patterns)
                    start_idx += n_patterns
                    
                    # Append patterns
                    diffraction_patterns.append(patterns)
                    patterns_found = True
                    print(f"  Row {i}: {patterns.shape}")
            
            if not patterns_found:
                raise ValueError(f"No diffraction patterns found in {master_file}")
            
            # Combine all patterns
            all_patterns = np.concatenate(diffraction_patterns, axis=0)
            print(f"\nTotal diffraction patterns: {all_patterns.shape}")
            
            return {
                'diffraction_patterns': all_patterns,
                'row_indices': row_indices
            }
            
    except Exception as e:
        print(f"Error reading diffraction data: {str(e)}")
        print(f"File path: {master_file}")
        # Return empty arrays if no data found
        return {
            'diffraction_patterns': np.array([]),
            'row_indices': {}
        }

def extract_parameters(config: Dict[str, str], scan_number: int) -> Dict[str, Any]:
    """
    Extract relevant parameters from the input files and peak analysis.
    """
    extracted_data = {}
    
    try:
        # Add cell information if available
        if 'cell_info' in config:
            print("\nExtracting cell information...")
            cell_info = extract_cell_info(config['cell_info'])
            if cell_info:
                for key, value in cell_info.items():
                    extracted_data[f'cell_{key}'] = value
                print(f"Added {len(cell_info)} cell parameters")
            else:
                print("Warning: No cell information was extracted")

        # Add peak analysis
        peak_filename = f'/net/micdata/data2/12IDC/ptychosaxs/peak_analysis/peak_analysis_scan_{scan_number}.csv'
        if os.path.exists(peak_filename):
            print(f"\nAnalyzing peak data from: {peak_filename}")
            peak_results = analyze_peak_data(
                filename=peak_filename,
                detector_distance=10000,  # Default SDD
                omega=0,  # Default omega
                xcenter=256,  # Default center
                ycenter=256   # Default center
            )
            
            # Add peak analysis results to extracted data
            extracted_data['peak_coords'] = peak_results['coords']
            extracted_data['peak_data'] = peak_results['peak_data']
            extracted_data['peak_figures'] = peak_results['figures']
            
            print(f"Successfully loaded peak analysis data for scan {scan_number}")
        else:
            print(f"Warning: Peak analysis file not found: {peak_filename}")

        # Add diffraction patterns
        print("\nExtracting diffraction patterns...")
        diffraction_data = extract_diffraction_data(config['diffraction_data'])
        if diffraction_data['diffraction_patterns'].size > 0:
            extracted_data['diffraction_patterns'] = diffraction_data['diffraction_patterns']
            extracted_data['diffraction_row_indices'] = diffraction_data['row_indices']
            print("Successfully extracted diffraction patterns")
        else:
            print("Warning: No diffraction patterns were found")

    except Exception as e:
        print(f"Error extracting parameters: {str(e)}")
        raise
    
    return extracted_data

def load_save_hdf5(config_file: str, 
                  output_file: str,
                  scan_number: int,
                  compression: str = 'gzip') -> Dict[str, np.ndarray]:
    """
    Load data from config file and save to a single HDF5 file.
    
    Parameters
    ----------
    config_file : str
        Path to JSON config file
    output_file : str
        Path to output HDF5 file
    scan_number : int
        Scan number to process
    compression : str, optional
        Compression type for HDF5 datasets
    """
    # Load configuration
    print(f"\nReading config file: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Update paths with scan number
    config = update_paths_with_scan(config, scan_number)
    
    # Print config contents
    print("\nConfig file contents with updated scan number:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Extract parameters
    print("\nExtracting parameters from files...")
    data = extract_parameters(config, scan_number)
    
    # Update output filename to include scan number if not specified
    if not output_file.endswith('.h5'):
        output_file = f"{output_file}_scan{scan_number}.h5"
    
    # Save to HDF5
    try:
        with h5py.File(output_file, 'w') as f:
            # Add each dataset to the file
            for key, value in data.items():
                if key == 'peak_figures':
                    # Convert figures to arrays and save
                    fig, (ax1, ax2, ax3) = value
                    fig.canvas.draw()
                    
                    # Create a group for figures
                    fig_group = f.create_group('peak_figures')
                    
                    # Convert and save the main figure
                    data_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                    fig_group.create_dataset('combined_plot', data=data_array, compression=compression)
                    
                    # Convert and save individual subplots
                    for i, ax in enumerate([ax1, ax2, ax3], 1):
                        subplot_fig = ax.get_figure()
                        for other_ax in subplot_fig.axes:
                            if other_ax != ax:
                                subplot_fig.delaxes(other_ax)
                        subplot_fig.canvas.draw()
                        
                        subplot_array = np.asarray(subplot_fig.canvas.buffer_rgba())[:, :, :3]
                        fig_group.create_dataset(f'subplot_{i}', data=subplot_array, compression=compression)
                    
                    print("Saved peak figures as arrays in HDF5 file")
                    continue
                
                elif key == 'peak_coords':  # Handle dictionary of coordinates
                    coord_group = f.create_group('peak_coords')
                    for coord_key, coord_value in value.items():
                        coord_group.create_dataset(coord_key, data=coord_value)
                elif isinstance(value, pd.DataFrame):  # Handle pandas DataFrames
                    f.create_dataset(key, data=value.to_numpy())
                elif np.isscalar(value):  # Handle scalar values
                    f.create_dataset(key, data=value)
                else:  # Handle array data
                    if compression:
                        f.create_dataset(key, data=value, compression=compression)
                    else:
                        f.create_dataset(key, data=value)
                    
            # Add metadata about data sources
            sources_group = f.create_group('_sources')
            for key, value in config.items():
                sources_group.attrs[key] = value
            
            # Add scan number to metadata
            sources_group.attrs['scan_number'] = scan_number
            
        print(f"\nSuccessfully saved data to: {output_file}")
        
    except Exception as e:
        print(f"\nError saving to HDF5 file {output_file}: {str(e)}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Load and save data for MIDAS analysis")
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON config file with data paths')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--scan', type=int, required=True,
                        help='Scan number to process')
    parser.add_argument('--compression', type=str, default='gzip',
                        choices=['gzip', 'lzf', 'none'],
                        help='Compression type for HDF5 datasets')
    
    args = parser.parse_args()
    
    compression = None if args.compression == 'none' else args.compression
    data = load_save_hdf5(args.input, args.output, args.scan, compression=compression)
    
    # Print summary of extracted data
    print("\nExtracted parameters:")
    for key, value in data.items():
        if key in ['peak_figures', 'scan_figure']:  # Skip figures in summary
            print(f"{key}: matplotlib Figure object")
        elif key == 'peak_coords':  # Handle coordinate dictionary
            print(f"{key}: dictionary with keys {list(value.keys())}")
        elif key == 'diffraction_row_indices':  # Handle row indices dictionary
            print(f"{key}: dictionary with {len(value)} rows")
        elif isinstance(value, pd.DataFrame):  # Handle DataFrames
            print(f"{key}: DataFrame with shape {value.shape}")
        elif isinstance(value, dict):  # Handle other dictionaries
            print(f"{key}: dictionary with keys {list(value.keys())}")
        elif np.isscalar(value):  # Handle scalar values
            print(f"{key}: {value}")
        else:  # Handle array data
            print(f"{key}: array shape {value.shape}")

if __name__ == "__main__":
    main() 