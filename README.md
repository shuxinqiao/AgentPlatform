# LangGraph Agent Framework

This demo used Django and React as backend and frontend. 

To make it running successfully, please check python and [node.js](https://nodejs.org/en) installation and version. 

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Integrate Your Tools](#integrate-your-tools)
- [API Overview](#api-overview)
- [File Structure](#file-structure)
- [DataBase Standard](#dataBase-standard)
    - [DataBase Structure](#database-structure)
    - [Data Type Registration](#data-type-registration)
    - [Tool Function Standard](#tool-function-standard)

## Requirements

python/pip related
- django == `5.1`
- channels == `4.1.0`
- daphne == `4.1.2`
- numpy == `1.26.3`
- langgraph == `0.2.2`
- pandas == `2.2.2`
- matplotlib == `3.8.4` (MUST, Plot 3D has different behaviour beyond this version)
- torch >= `2.2.2`

node.js == `20.16.0`

ChatGPT API, ask me if needed.

## Setup

```sh
create a file under root dir with name `.env`, write API into it.
```

Start Django (both core and NN models)
port is `http:\\localhost:8000\\`

```sh
$ python manage.py runserver
```

Start frontend, port is `http:\\localhost:5173\\` (access it to see UI)

```sh
$ cd frontend

$ npm install
$ npm run dev
```

## Integrate Your Tools

Please follow `/langgraph/tools/GRUFNO_tools.py` and add tools in this directory only.

Then register it in `/langgraph/graph.py` `define_tools = [GRUFNO_Prediction, plot_3d_image, ...]`.

Check your Git push files, avoid personal and large file to keep project clean.


## API Overview:

`model/grufno/sat/<userID>/*` for GRU-FNO model access.

`db/<userID>/*` for user database access.

`temp/<userID>/<filename>/` for user temporary files access.


## File Structure

```
Django_Next_Langgraph_Stream (Django Project)
│
├── Django_Next_Langgraph_Stream (Django Core App)
│
├── frontend (React Frontend UI)
│
├── grufno (Django App - GRU-FNO model)
|
├── langgraph_stream (Django App - Langgraph Streaming)
│   └── Tools:
│       ├── GRUFNO_tools.py
│
├── .env (ChatGPT API key)
```


## DataBase Standard

Using sqlite3 now for simplicity in early stage.

### DataBase Structure:
```
Running_DB.db/
│                  
├── User_1_table
├── User_2_table   
├── ...            
└── User_N_table ──┬── rowid, data BLOB,   type TEXT, description TEXT   
                   ├──     1,    binary, pddataframe,          ABCDEFG
                   ├──     2,    binary,     nparray,          AEFs5as
                   ├──     3,    binary,        dict,          a4Q1sxz
                   ├──    ...
                   └──     K,    binary,        ????,          ???????
```

### Data Type Registration

To allow sqlite3 automatically retrieve binary data into structured,
each type in Table should be defined inside ```./db_utils.py``` with similar format as following:

```python
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
```

### Tool Function Standard

Tool function should obtain ID from graph state: ```state = {..., "user_id": 123, ...}```.

Also, get data by obtain "rowid (int)" from LLM tool call.

```python
get_data_sqlite3(filename="Running_DB.db", table=state["user_id"], id=rowid, type="pddataframe")
store_data_sqlite3(filename="Running_DB.db", table=state["user_id"], type="nparray", data=data, description=desciption)
```
