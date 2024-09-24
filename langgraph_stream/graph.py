from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode

# Utility libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# Tools Registration
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langgraph_stream.db_utils import *
from langgraph_stream.tools.random_array_plot import random_array_plot
from langgraph_stream.tools.output_file_tools import data_to_file
from langgraph_stream.tools.GRUFNO_tools import GRUFNO_Prediction_with_predefined_settings, GRUFNO_Prediction_with_direct_input, plot_3d_image
from langgraph_stream.tools.CMG_data_parser import CMG_data_parser_to_co2_input


def prepare(llm, userID, prompt):

    # Init Database if not exists
    table_check(db_name="test.db", table_name=userID)

    def insert_user_input(filename, userID):
        data = pd.read_csv(filename)

        # Get User Input Data details for us store in Database
        shape = data.shape
        column_names = list(data.columns)
        description = f"DataFrame with filename=\"{filename}\" created by \"User Input\" with length={shape[0]} and columns={column_names[1:]}"
        print(f"Description: \n\t{description}\n")

        # Store Data into Database
        store_data_sqlite3(filename="test.db", table=userID, data=data, type="pddataframe", description=description)

    files = ["Duvernay_test_clean", "Duvernay_test", "Duvernay_train_clean", "Duvernay_train", ]
    for filename in files:
        #insert_user_input(f"E:\\Django_Next_Langgraph_Stream\\langgraph_stream\\geologist_test_data\\{filename}.csv", userID)
        pass

    repl = PythonREPL()
    @tool
    def python_repl(
        code: Annotated[str, "The python code to execute to generate your dagram."],
    ):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
        )

    ###############################
    # Register Tools here
    ###############################
    define_tools = [GRUFNO_Prediction_with_predefined_settings, GRUFNO_Prediction_with_direct_input, plot_3d_image, CMG_data_parser_to_co2_input]

    llm_with_tools = prompt | llm.bind_tools(define_tools)

    from langchain_core.messages import ToolMessage

    class BasicToolNode:
        """A node that runs the tools requested in the last AIMessage."""

        def __init__(self, tools: list) -> None:
            self.tools_by_name = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            outputs = []
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}
        
    # build the agent
    class State(TypedDict):
        # Messages have the type "list". The `add_messages` function
        # in the annotation defines how this state key should be updated
        # (in this case, it appends messages to the list, rather than overwriting them)
        messages: Annotated[list, add_messages]
        userID: str

    # add a chatbot node
    graph_builder = StateGraph(State)

    def chatbot(state: State):
        db_info = get_db_json(db_name='test.db', table_name=state["userID"])
        
        state = {
            "messages": [db_info] + state["messages"],
            "userID": userID,
        }

        # return {"messages": [llm_with_tools.invoke([] + state["messages"])]}
        #return {"messages": [llm_with_tools.invoke([db_info] + state["messages"])], "userID": state["userID"]}

        state["messages"] = [llm_with_tools.invoke(state["messages"])]

        return state
    
    graph_builder.add_node("chatbot", chatbot)

    # add a tool node
    #tool_node = BasicToolNode(tools=define_tools)
    tool_node = ToolNode(define_tools)
    graph_builder.add_node("tools", tool_node)

    from typing import Literal

    def route_tools(
        state: State,
    ) -> Literal["tools", "__end__"]:
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "__end__"

    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile(checkpointer=memory)
    


    return graph