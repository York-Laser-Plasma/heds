# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:29:48 2024
Example usage of the FLASH data analysis functions

@author: yuyao
"""
#%%
from FLASH_Data_Reader import *
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt

# User only needs to change this variable to switch fields
selected_field = 'dens'  # Options could be 'dens', 'tion', 'tele', 'trad', 'pressure', etc.

# Map each field to its correct unit
field_units = {
    'dens': 'code_mass/code_length**3',
    'tion': 'code_temperature',
    'tele': 'code_temperature',
    'trad': 'code_temperature',
    'pres': 'dyn/cm**2'
    # Add other fields and units as needed
}

file_dir = 'data'
file_sub_dir = 'transparent3' # 'SedovChamberTest'
filename_pattern = 'radslab_hdf5_plt_cnt_????'
save_path = os.path.join(file_dir, file_sub_dir, "post-processing")

# Load and process the data
data = load_and_process_data(file_dir, file_sub_dir, filename_pattern, lrefine_max=6)

# Extract the field array for the selected field
field_unit = field_units[selected_field]
field_array = extract_field_array(data, field=selected_field, unit=field_unit)
#%%
# Example: 1D line plot
fig, ax = plot_1D_line(
    data, 
    field_array, 
    time_step=31, 
    line_index=0, 
    axis="y", 
    title=f"{selected_field.capitalize()} Along X"
)
#%%
# Example: 2D map
fig, ax = plot_2D_map(
    figsize=(8, 4),
    data=data, 
    plotvar=field_array, 
    time_step=30, 
    log_scale=True, 
    cmap='OrRd',
    title=f"{selected_field.capitalize()}"
)
#%%
# Example: temporal evolution
fig, ax = plot_temporal_evolution(
    data, 
    field_array, 
    xpos=0, 
    ypos=6000, 
    title=f"{selected_field.capitalize()} Temporal Evolution"
)
#%%
# Example: animation of 2D map evolution
anim = animate_2D_map(
    data=data,
    field_array=field_array,
    log_scale=True,
    cmap="OrRd",
    title=f"{selected_field.capitalize()} Evolution",
    interval=100,
    save_as="1.gif"
)

#%%
save_all_frames_as_png(
    data=data,
    field_array=field_array,
    field_name=selected_field,
    output_dir="output_images",
    vmax = 10.0,
    vmin = 1.0e-5,
    cmap="OrRd",
    log_scale=True,
    title=f"{selected_field.capitalize()} Evolution"
)

