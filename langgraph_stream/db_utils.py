# Sqlite3 register 
import sqlite3
import io

import numpy as np
import pandas as pd
import pickle
import json


#################################################
# Registry Functions
#################################################
# Global V
CONVERTER_FUNCTINOS = dict()

# Numpy Array
def nparray_in(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def nparray_out(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Register the adapter and converter for Numpy Array
sqlite3.register_adapter(np.ndarray, nparray_in)
sqlite3.register_converter("nparray", nparray_out)
CONVERTER_FUNCTINOS["nparray_out"] = nparray_out

# Pandas DataFrame
def pddataframe_in(df):
    out = io.BytesIO()
    df.to_pickle(out)
    out.seek(0)
    return sqlite3.Binary(out.read())

def pddataframe_out(binary):
    out = io.BytesIO(binary)
    out.seek(0)
    return pd.read_pickle(out)

# Register the adapter and converter for Pandas DataFrame
sqlite3.register_adapter(pd.DataFrame, pddataframe_in)
sqlite3.register_converter("pddataframe", pddataframe_out)
CONVERTER_FUNCTINOS["pddataframe_out"] = pddataframe_out


# Python Default Dictionary
def dict_in(data):
    return pickle.dumps(data)

def dict_out(blob):
    return pickle.loads(blob)

sqlite3.register_adapter(dict, dict_in)
sqlite3.register_converter("dict", dict_out)
CONVERTER_FUNCTINOS["dict_out"] = dict_out




#################################################
# Store and Retrieve Functions
#################################################
def store_data_sqlite3(filename=":memory:", table=None, data=None, type=None, description=None):
    assert table != None, "No arg 'table' provided for 'store_data_sqlite3'"
    assert type != None, "No arg 'col' provided for 'store_data_sqlite3'"

    con = sqlite3.connect(filename, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    # Insert the value into the specified table and column
    cur.execute(f"INSERT INTO {table} (data, type, description) VALUES (?, ?, ?);", (data, type, description))
    rowid = cur.lastrowid

    # Commit the transaction
    con.commit()
    con.close()

    return rowid

def get_data_sqlite3(filename=":memory:", table=None, id=None, type=None):
    assert table != None, "No 'table' provided for 'get_data_sqlite3'"
    assert id != None, "No 'id' provided for 'get_data_sqlite3'"
    assert type != None, "No arg 'type' provided for 'get_data_sqlite3'"

    con = sqlite3.connect(filename, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute(f"SELECT data, type from {table} WHERE rowid = ?", (id,))
    output = cur.fetchone()

    # Close Connection
    con.close()

    # Request Type Check
    if type != output[1]:
        raise TypeError(f"Tool function requests type={type} but {output[1]} is given.")
    else:
        return CONVERTER_FUNCTINOS[f"{type}_out"](output[0])


#################################################
# Table Creation Functions
#################################################
def table_check(db_name, table_name, columns={"data": "BLOB", "type": "TEXT", "description": "TEXT"}):
    """
    Create a table in the SQLite database if it doesn't exist.

    Args:
        db_name (str): The name of the database file.
        table_name (str): The name of the table.
    
    Returns:
        None
    """
    # Numpy array columns
    #columns = {"array": "nparray", "description": "TEXT"}
    # Connect to the database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    table_exists = cursor.fetchone()


    # If table doesn't exist, create it
    if not table_exists:
        columns_str = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
        create_table_query = f"CREATE TABLE {table_name} ({columns_str});"
        cursor.execute(create_table_query)
        conn.commit()
    else:
        # Check for missing columns
        cursor.execute(f"PRAGMA table_info({table_name});")
        existing_columns = {row[1]: row[2] for row in cursor.fetchall()}
        for col, dtype in columns.items():
            if col not in existing_columns:
                alter_table_query = f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype};"
                cursor.execute(alter_table_query)
                conn.commit()
    
    # Close the connection
    conn.close()


#################################################
# Description JSON Return Functions
#################################################
def get_db_json(db_name='test.db', table_name="test", columns_to_exclude=["data", ]):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Get all column names from the table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]

    # Filter out the columns to exclude
    columns_to_select = ['rowid'] + [col for col in columns if col not in columns_to_exclude]

    # Construct the SQL query
    query = f"SELECT {', '.join(columns_to_select)} FROM {table_name}"

    # Execute the query and fetch results
    cursor.execute(query)
    rows = cursor.fetchall()

    # Get column names from the results
    column_names = columns_to_select

    # Convert the rows to a list of dictionaries
    results = [dict(zip(column_names, row)) for row in rows]

    # Convert the results to JSON format
    json_results = json.dumps(results)

    # Close the connection
    conn.close()

    return json_results
