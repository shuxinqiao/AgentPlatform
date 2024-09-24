from langchain_core.tools import tool
from typing import Annotated, List, Tuple
from langgraph.prebuilt import InjectedState
from langgraph_stream.db_utils import *
import json
import os

import langgraph_stream.tools.flowgrid_new as fg
import numpy as np


@tool
def CMG_data_parser_to_co2_input(
    userID: Annotated[str, InjectedState("userID")],
    rowid: Annotated[str, f"User uploaded CMG .dat file location."],
):
    """
    CMG data file (.dat) parser.
    It parses data into CO2 injection Model acceptable input format.
    In most of times, this .dat file should be provided by user, except user indicated some public sources that could be reach.
    Output: numpy array in shape (Batch, time, Channel, Z, X, Y), it is not ready to use, need to determine injection location, time steps and inject rates.
    """
    
    data_path = get_data_sqlite3(filename="test.db", table=userID, id=rowid, type="userinput")

    # Define input and output file names
    dat_file = data_path

    # Construct a single grid from the dat file
    grid = fg.CMG()
    grid.CORNER(dat_file, ['CORNERS'])
    X_blocks, Y_blocks, Z_blocks = grid.size[0], grid.size[1], grid.size[2]

    # Initialize Rate and Bhp arrays


    # Get well data from the file
    wells = grid.get_wells(dat_file)
    data = wells.wells
    Rate_list = []
    Bhp_list = []
    Rate_array = np.zeros((X_blocks, Y_blocks))
    Bhp_array = np.zeros((X_blocks, Y_blocks))

    # Iterate through data and fill the arrays
    for entry in data:
        if entry.get('TYPE') == 'INJ':
            loc = entry['LOC']
            Rate_val = float(entry['CON_VAL'][0])
            Bhp_val = float(entry['CON_VAL'][1])

            Rate_array[loc[0], loc[1]] = Rate_val
            Bhp_array[loc[0], loc[1]] = Bhp_val

            
    Rate_list.append(Rate_array)
    Bhp_list.append(Bhp_array)


    # Transform Rate_array
    # Rate_all = np.expand_dims(Rate_list, axis=(0, -1))
    # Bhp_all = np.expand_dims(Bhp_list, axis=(0, -1))

    Rate_all = np.expand_dims(np.stack(Rate_list), axis=-1)
    Bhp_all = np.expand_dims(np.stack(Bhp_list), axis=-1)

    Rate_transformed = np.transpose(Rate_all, (0, 3, 2, 1))
    Rate_transformed = np.expand_dims(Rate_transformed, axis=(2, 3))
    Rate_transformed = np.repeat(Rate_transformed, 1, axis=1)
    Rate_transformed = np.repeat(Rate_transformed, Z_blocks, axis=3)
    Rate_transformed *= 1e-6  # Scale

    # Transform Bhp_array
    Bhp_transformed = np.transpose(Bhp_all, (0, 3, 2, 1))
    Bhp_transformed = np.expand_dims(Bhp_transformed, axis=(2, 3))
    Bhp_transformed = np.repeat(Bhp_transformed, 1, axis=1)
    Bhp_transformed = np.repeat(Bhp_transformed, Z_blocks, axis=3)
    Bhp_transformed[...] = np.max(Bhp_transformed)
    Bhp_transformed *= 1e-5  # Scale

    # Read and transform Perm and Por from the dat file
    Perm_transformed = []
    Por_transformed = []

    for prop_name in ['PERMI', 'POR']:
        prop = grid.read_prop(dat_file, prop_name, add=True)
        prop = np.expand_dims(prop, axis=(0, -1))
        prop = np.transpose(prop, (0, 4, 1, 2, 3))
        prop = np.expand_dims(prop, axis=1)
        prop = np.repeat(prop, 1, axis=1)

        # Store the transformed property
        if prop_name == 'PERMI':
            Perm_transformed = prop
        else:
            Por_transformed = prop

            
    bound = grid.bound_type(dat_file)
    # print(bound.shape)
    # Merge all properties into a single array
    comb_input = np.concatenate([Perm_transformed, Por_transformed, Rate_transformed, Bhp_transformed,bound], axis=2)

    # Save merged array
    # np.save('merged_properties.npy', merged_array)

    # Print the shape of the merged array
    #print(comb_input.shape)
    
    description = f"Parsed by function 'CMG_data_parser_to_co2_input' and user uploaded file with rowid={rowid}. The output shape is {comb_input.shape}."
    new_file_rowid = store_data_sqlite3(filename="test.db", table=userID, data=comb_input, type="nparray", description=description)

    return json.dumps(f"User uploaded CMG .dat file with rowid={rowid} has been parsed as a numpy array into database with rowid={new_file_rowid} and shape {comb_input.shape}.")

