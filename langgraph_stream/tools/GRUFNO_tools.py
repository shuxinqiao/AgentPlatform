from langchain_core.tools import tool
import json

from typing import Annotated, List, Tuple
from langgraph.prebuilt import InjectedState
from langgraph_stream.db_utils import *
import requests


@tool
def GRUFNO_Prediction_with_predefined_settings(
    userID: Annotated[str, InjectedState("userID")],
    type: Annotated[str, "'sat' for saturation."],
    percentile: Annotated[str, f"must in list {["P20", "P40", "P60", "P80", "P30", "P50", "P25", "P75", "P5", "P10", "P15", "P65", "P95"]}."],
    location: Annotated[List[List[int]],"Inside List MUST be length=2, each value MUST in [0,60)."],
    rate: Annotated[List[float],"MUST be same length as param::location, float value MUSt in [0,2]."],
    bhp: Annotated[int,"Any value in [36000,75000]."],
    boundary: Annotated[bool,"True means using leaky boundary condition, False means No leakage boundary condition."],
):
    """
    CO2 Saturation or Pressure prediction with output size (Time=10, Z=24, X=60, Y=60).
    This function should be used only if user specifically said they don't have any direct geology information as input.
    """
    API_url = "http://localhost:8000/model/grufno/sat/" + userID

    params = {
        'percentile': json.dumps(percentile),
        'location': json.dumps(location),
        'rate': json.dumps(rate),
        'bhp': json.dumps(bhp),
        'boundary': json.dumps(boundary)
    }

    response = requests.get(API_url, params=params)

    if response.status_code == 200:
        data = response.json()
        rowid = data.get('rowid')
        return json.dumps(f"Prediction has been produced with rowid={rowid} in database.")
    else:
        return json.dumps(f"Model communication failed with status code {response.status_code}")


@tool
def GRUFNO_Prediction_with_direct_input(
    userID: Annotated[str, InjectedState("userID")],
    type: Annotated[str, "'sat' for saturation."],
    rowid: Annotated[str, f"User uploaded data rowid in database, type of that record must be 'nparray'."],
    location: Annotated[List[List[int]],"Inside List MUST be length=2, each value MUST in [0,60)."],
    rate: Annotated[List[float],"MUST be same length as param::location, float value MUSt in [0,2]."],
    bhp: Annotated[int,"Any value in [36000,75000]."],
    boundary: Annotated[bool,"True means using leaky boundary condition, False means No leakage boundary condition."],
):
    """
    CO2 Saturation or Pressure prediction with output size (Time=10, Z=24, X=60, Y=60).
    This function should be used if user has their own direct geology data as input.
    """
    API_url = "http://localhost:8000/model/grufno/direct/sat/" + userID

    params = {
        'rowid': json.dumps(rowid),
        'location': json.dumps(location),
        'rate': json.dumps(rate),
        'bhp': json.dumps(bhp),
        'boundary': json.dumps(boundary)
    }

    response = requests.get(API_url, params=params)

    if response.status_code == 200:
        data = response.json()
        rowid = data.get('rowid')
        return json.dumps(f"Prediction has been produced with rowid={rowid} in database.")
    else:
        return json.dumps(f"Model communication failed with status code {response.status_code}")



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from django.conf import settings
import os
import re

@tool
def plot_3d_image(
    userID: Annotated[str, InjectedState("userID")],
    rowid: Annotated[int, "rowid of data that requires visulization in database"],
    batch: Annotated[int, "Index of the batch dimension of required data."]=0,
    time: Annotated[int, "Index of the time dimension of required data."]=0,
    channel: Annotated[int, "the channel "]=0, 
    show_legend: Annotated[bool, "Whether to show the color gradient legend."]=True, 
    value_range: Annotated[List[float], "List of minimum and maximum values to display. If None, the entire range of values will be displayed."]=None, 
    figsize: Annotated[List[int], "Width and height of the figure in inches."]=(8,6), 
    dpi: Annotated[int, "Dots per inch (resolution) of the figure."]=100
    ):
    """
    Plot a 3D image from data with specified batch, time, and channel.
    Find the valid range for some parameters by checking info of that row in database.
    Data with given rowid MUST in shape (Batch, Time, Channel, Z, X, Y).
    """
    # Extract the specified data slice
    data = get_data_sqlite3(filename="test.db", table=userID, id=rowid, type="nparray")
    slice_data = data[batch, time, channel]
    
    # Determine value range for transparency
    if value_range is None:
        value_min = np.min(slice_data)
        value_max = np.max(slice_data)
    else:
        value_min, value_max = value_range
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid representing voxel indices
    z, x, y = np.indices(tuple(x+1 for x in slice_data.shape))
    z = z[::-1]
    
    # Normalize data to map to colors
    norm = Normalize(vmin=np.min(slice_data), vmax=np.max(slice_data))
    
    # Mask values outside the specified range for transparency
    masked_data = np.where((slice_data >= value_min) & (slice_data <= value_max), slice_data, 0)            

    # Plot the voxels with gradient color and transparency
    ax.voxels(x, y, z, masked_data, facecolors=plt.cm.autumn(norm(slice_data)), edgecolors=None, shade=False)    
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Image (Batch: {batch}, Time: {time}, Channel: {channel})')
    
    # Add color legend if specified
    if show_legend:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.autumn, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Value')
    
    # Save the plot
    filename = None
    dir_path = os.path.join(settings.TEMP_ROOT, userID)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        filename = 0  # No files, return 0
    
    # Regex pattern to match filenames with numbers (e.g., '1.py', '31.jsx')
    pattern = re.compile(r"^(\d+)\..+$")
    
    max_number = 0
    for filename in os.listdir(dir_path):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)

    save_path = f"/temp/{userID}/{str(max_number + 1) + '.png'}"
    plt.savefig(os.path.join(dir_path, str(max_number + 1) + '.png'))
    
    return json.dumps(f"3D plot has been saved on path={'http://localhost:5173' + save_path}, (return path in markdown image format).")
    #plt.show()

