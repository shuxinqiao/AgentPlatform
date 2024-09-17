from typing import Annotated, List
import pandas as pd
import numpy as np

from langchain_core.tools import tool
from langgraph_stream.db_utils import *
import json
import os



@tool
def data_to_file(
    file_id: Annotated[int, "Rowid of requested data in Database."],
    data_type: Annotated[str, "Type of requested data."],
    output_file_type: Annotated[str, "Type of the requested output file, must be one of ['csv']."] = "csv",
):
    """
    Retrieve data from database into requested type.
    """
    try:
        data = get_data_sqlite3(filename="test.db", table="test", id=file_id, type=data_type)
        output_dir_path = os.getcwd() + "/user_output_files/"
        os.makedirs(output_dir_path, exist_ok=True)  # Create the directory if it doesn't exist
        files = [f for f in os.listdir(output_dir_path) if os.path.isfile(os.path.join(output_dir_path, f))]
        
        sorted_files = sorted(files)    # Sort the files in ascending order
        if sorted_files != []:
            index = int(sorted_files[-1].split('.')[0])
        else:
            index = 0

        match output_file_type:
            case "csv":
                output_path = output_dir_path + f'{str(index + 1)}.csv'
                data.to_csv(output_path, index=False)
        
        return f"Data has been saved as {output_file_type} with path:{output_path}."

        

    except Exception as e:
        # Handle any exceptions and return an error message
        return json.dumps({"data_to_file function error": str(e)})