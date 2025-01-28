#!/usr/bin/env python3
"""
coordinate_convert_midas.py

Functions for converting detector coordinates to lab frame and calculating angles.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, List

def transform_coordinates(xorig: np.ndarray,
                        yorig: np.ndarray,
                        omega: float,
                        detector_distance: float,
                        xcenter: float = 0,
                        ycenter: float = 0) -> Dict[str, np.ndarray]:
    """
    Transform detector coordinates to lab frame and calculate angles
    
    Parameters
    ----------
    xorig : array_like
        Original x coordinates on detector
    yorig : array_like
        Original y coordinates on detector
    omega : float
        Sample rotation angle in radians
    detector_distance : float
        Sample to detector distance in same units as x,y
    xcenter : float
        X coordinate of beam center
    ycenter : float
        Y coordinate of beam center
        
    Returns
    -------
    dict
        Dictionary containing:
        - XLab, YLab, ZLab: Lab frame coordinates
        - two_theta: Scattering angle
        - eta: Azimuthal angle
        - chi: Sample rotation angle
    """
    # Center coordinates
    x = xorig - xcenter
    y = yorig - ycenter
    
    # Calculate r in detector plane
    r = np.sqrt(x**2 + y**2)
    
    # Calculate angles
    two_theta = np.arctan2(r, detector_distance)
    eta = np.arctan2(y, x)
    
    # Calculate lab frame coordinates
    XLab = detector_distance * np.sin(two_theta) * np.cos(eta)
    YLab = detector_distance * np.sin(two_theta) * np.sin(eta)
    ZLab = detector_distance * np.cos(two_theta)
    
    return {
        'XLab': XLab,
        'YLab': YLab,
        'ZLab': ZLab,
        'two_theta': two_theta,
        'eta': eta,
        'chi': np.full_like(XLab, omega)
    }

def plot_points(coords: Dict[str, np.ndarray],
               intensities: np.ndarray = None,
               title: str = '') -> None:
    """
    Plot transformed coordinates in 3D
    
    Parameters
    ----------
    coords : dict
        Dictionary of coordinates from transform_coordinates()
    intensities : array_like, optional
        Point intensities for coloring
    title : str, optional
        Plot title
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if intensities is None:
        intensities = np.ones_like(coords['XLab'])
    
    scatter = ax.scatter(coords['XLab'],
                        coords['YLab'],
                        coords['ZLab'],
                        c=intensities,
                        cmap='viridis')
    
    plt.colorbar(scatter)
    ax.set_xlabel('X Lab')
    ax.set_ylabel('Y Lab')
    ax.set_zlabel('Z Lab')
    ax.set_title(title)
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    x = np.array([100, 200, 300])
    y = np.array([150, 250, 350])
    
    coords = transform_coordinates(
        xorig=x,
        yorig=y,
        omega=0,
        detector_distance=1000,
        xcenter=256,
        ycenter=256
    )
    
    plot_points(coords, title='Example Transform') 