# %%
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# %%
from fastapi import FastAPI
from langserve import add_routes
# from langserve.pydantic_v1 import BaseModel, Field

# %%
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

api_key = os.getenv("OPENAI_API_KEY")


# %%
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOpenAI(
    openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0, streaming=True
)


def chatbot(state: State):
    # print(state["messages"])
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
mygraph = graph_builder.compile()
# output = mygraph.invoke({"messages": ["Hello, how are you?"]})
# print(output["messages"])

# %%
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    mygraph,
    path="/chat",
)

# %%
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
