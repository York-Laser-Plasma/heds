"""
Helios Data Plotting Utilities

Author: Xu Zhao
Affiliation: York Plasma Institute, School of Physics, Engineering and Technology,
             University of York, York YO10 5DD, United Kingdom
Email: xu.zhao@york.ac.uk
Last Updated: 2023-11-14

Description:
This module provides functions for loading and processing Helios data, as well as 
plotting functions for visualizing density, electron temperature, and radius evolution.
Each function is designed to produce publication-quality figures with consistent style.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_data(file_path):
    """
    Loads and processes data from the specified Helios file.

    Parameters:
    - file_path: str, path to the data file.

    Returns:
    - dict: A dictionary containing processed data arrays for time, radius, density, 
            electron density, temperature, pressure, velocity, and volume.
    """
    data = xr.open_dataset(file_path)

    # Extract variables and convert to numpy arrays
    time_whole = data['time_whole'].values
    zone_boundaries = data['zone_boundaries'].values
    mass_density = data['mass_density'].values
    elec_density = data['elec_density'].values
    ion_temperature = data['ion_temperature'].values
    elec_temperature = data['elec_temperature'].values
    rad_temperature = data['radiation_temperature'].values
    zone_mass = data['zone_mass'].values
    pressure = data['ion_pressure'].values + data['elec_pressure'].values
    fluid_velocity = data['fluid_velocity'].values / 100000  # Convert to km/s
    volume = zone_mass / mass_density

    # Calculate time edges for pcolormesh
    time_diff = np.diff(time_whole) / 2
    time_edges = np.concatenate(([time_whole[0] - time_diff[0]], time_whole[:-1] + time_diff, [time_whole[-1] + time_diff[-1]]))

    # Calculate radius edges for pcolormesh
    radius_diff = np.diff(zone_boundaries, axis=0) / 2
    radius_edges = np.vstack((zone_boundaries[0, :] - radius_diff[0, :],
                              zone_boundaries[:-1, :] + radius_diff,
                              zone_boundaries[-1, :] + radius_diff[-1, :]))

    return {
        "time_whole": time_whole,
        "zone_boundaries": zone_boundaries,
        "mass_density": mass_density,
        "elec_density": elec_density,
        "ion_temperature": ion_temperature,
        "elec_temperature": elec_temperature,
        "rad_temperature": rad_temperature,
        "zone_mass": zone_mass,
        "pressure": pressure,
        "fluid_velocity": fluid_velocity,
        "volume": volume,
        "time_edges": time_edges,
        "radius_edges": radius_edges
    }

def plot_radius_evolution(time, radius, figsize=(8.5 / 2.54, 8.5 / 1.618 / 2.54), line_width=0.5,
                          line_color='black', font_size=7, font_family='Arial',
                          dpi=200, border_width=0.5, tick_length=3, tick_width=0.5):
    """
    Plots the evolution of radius over time for each zone.

    Parameters:
    - time: 1D array-like, time points.
    - radius: 2D array-like, radius values for each time and zone.
    - figsize: tuple, figure size in inches, default (8.5 cm, 6.5 cm).
    - line_width: float, line width for the plot.
    - line_color: str, line color.
    - font_size: int, font size for labels and title.
    - font_family: str, font family for labels and title.
    - dpi: int, resolution of the figure in dots per inch.
    - border_width: float, width of the figure border.
    - tick_length: float, length of ticks on both axes.
    - tick_width: float, width of ticks on both axes.

    Returns:
    - ax: Matplotlib Axes object, for further customization.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(radius.shape[1]):
        ax.plot(time, radius[:, i], color=line_color, lw=line_width)

    ax.set_xlabel("Time (ns)", fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel(r"Radius ($\rm{\mu}$m)", fontsize=font_size, fontfamily=font_family)
    ax.set_title("Radius Evolution Over Time for Each Zone", fontsize=font_size, fontfamily=font_family)
    ax.tick_params(axis='both', which='major', labelsize=font_size, length=tick_length, width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)
    return ax

def plot_density_pcolormesh(time_edges, radius_edges, density, figsize=(8.5 / 2.54, 8.5 / 1.618 / 2.54),
                            cmap='jet', font_size=7, font_family='Arial',
                            dpi=200, border_width=0.5, tick_length=3, tick_width=0.5):
    """
    Plots a density pcolormesh with a color bar.

    Parameters:
    - time_edges: 1D array-like, edges for the time axis (scaled to ns).
    - radius_edges: 1D or 2D array-like, edges for the radius axis (scaled to microns).
    - density: 2D array-like, density values for each time and radius.
    - figsize, cmap, font_size, font_family, dpi, border_width, tick_length, tick_width.

    Returns:
    - ax: Matplotlib Axes object.
    """

    # Convert figsize from cm to inches
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot pcolormesh for density data
    cmesh = ax.pcolormesh(time_edges, radius_edges.T, density.T, shading='auto', cmap=cmap)

    # Add color bar
    cbar = fig.colorbar(cmesh, ax=ax)
    cbar.ax.tick_params(labelsize=font_size, length=tick_length, width=tick_width)
    cbar.outline.set_linewidth(border_width)  # Set colorbar border thickness

    # Set colorbar label at the top and horizontal
    cbar_label = r"$\rho$ (g/cc)"
    cbar.ax.text(0.5, 1.02, cbar_label, ha='center', va='bottom',
                 fontsize=font_size, fontfamily=font_family, transform=cbar.ax.transAxes)

    # Set labels and title
    ax.set_xlabel("Time (ns)", fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel(r"Radius ($\rm{\mu}$m)", fontsize=font_size, fontfamily=font_family)
    ax.set_title("Density Plot", fontsize=font_size, fontfamily=font_family)

    # Set font size and family for tick labels
    ax.tick_params(axis='both', which='major', labelsize=font_size, length=tick_length, width=tick_width)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(font_family)

    # Set border width for all spines
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    # Return the Axes object for further customization
    return ax

def plot_electron_temp_pcolormesh(time_edges, radius_edges, elec_temp, figsize=(8.5 / 2.54, 8.5 / 1.618 / 2.54),
                                  cmap='jet', font_size=7, font_family='Arial',
                                  dpi=200, border_width=0.5, tick_length=3, tick_width=0.5):
    """
    Plots an electron temperature (Te) pcolormesh with a color bar.

    Parameters:
    - time_edges, radius_edges, elec_temp, figsize, cmap, font_size, font_family, dpi, border_width, tick_length, tick_width.

    Returns:
    - ax: Matplotlib Axes object.
    """
    # Convert figsize from cm to inches
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot pcolormesh for electron temperature data
    cmesh = ax.pcolormesh(time_edges, radius_edges.T, elec_temp.T, shading='auto', cmap=cmap, )

    # Add color bar
    cbar = fig.colorbar(cmesh, ax=ax)
    cbar.ax.tick_params(labelsize=font_size, length=tick_length, width=tick_width)
    cbar.outline.set_linewidth(border_width)  # Set colorbar border thickness

    # Set colorbar label at the top and horizontal
    cbar_label = r"$T_e$ (eV)"
    cbar.ax.text(0.5, 1.02, cbar_label, ha='center', va='bottom',
                 fontsize=font_size, fontfamily=font_family, transform=cbar.ax.transAxes)

    # Set labels and title
    ax.set_xlabel("Time (ns)", fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel(r"Radius ($\rm{\mu}$m)", fontsize=font_size, fontfamily=font_family)
    ax.set_title("Electron Temperature Contour", fontsize=font_size, fontfamily=font_family)

    # Set font size and family for tick labels
    ax.tick_params(axis='both', which='major', labelsize=font_size, length=tick_length, width=tick_width)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(font_family)

    # Set border width for all spines
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    # Return the Axes object for further customization
    return ax