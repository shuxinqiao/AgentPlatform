from langchain_core.tools import tool
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re

from typing import Annotated
from langgraph.prebuilt import InjectedState

from langgraph_stream.db_utils import *

from django.conf import settings


@tool
def random_array_plot(
    length: Annotated[int, "1D array length."],
    userID: Annotated[str, InjectedState("userID")],
):
    """
    Generates a random 1D numpy array and its line plot.
    """
    #userID = state['userID']
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
 
    x = np.arange(length)
    y = np.random.rand(length)

    description = f"Numpy array created by function \'random_array_plot\' with shape={y.shape}."
    rowid = store_data_sqlite3(filename="test.db", table=userID, data=y, type='nparray', description=description)

    # Plot the data
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Line Plot of Random Data')

    # Save the plot
    save_path = f"/temp/{userID}/{str(max_number + 1) + '.png'}"
    plt.savefig(os.path.join(dir_path, str(max_number + 1) + '.png'))

    return json.dumps(f"Generated 1D numpy array has been stored as rowid={rowid} and plot has been saved on path={'http://localhost:5173' + save_path}, (return this as markdown image format).")