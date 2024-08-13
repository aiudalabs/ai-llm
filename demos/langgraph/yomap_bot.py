# Import Libaries
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

import datetime

from langgraph.prebuilt import ToolNode

# Firebase
from firebase_admin import initialize_app, firestore, credentials

cred = credentials.Certificate("goblob-95e2a-1e236e39de6c.json")
app = initialize_app(credential=cred)
db = firestore.client()

# API keys
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

api_key = os.getenv("OPENAI_API_KEY")


# LangGraph Utils
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# Tools
# Functions Implementation
@tool
def fetch_user_information():
    """Fetch the current user info based on the displayName of the user.
    Use this tools at the begining to know the user information and
    interact with him/her using the real name.
    """

    config = ensure_config()  # Fetch from the context
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)


@tool
def get_service_provider_from_firebase(tag: str):
    """Get all the service providers based on the tag
    or the name of the category. Use this tool each time the user
    request some information about a particular service.

    Example: if the user ask about spa, then you use this tool with tag = spa
    """

    profile = db.collection("profiles")
    print(tag)
    docs = profile.where("service.text", "==", tag).get()
    return [doc.to_dict()["displayName"] for doc in docs]


@tool
def get_profile_info_from_firebase(name: str):
    """Get profile info based on the name of the service provider.
    Use this tool when the user want to know more details about
    a particular service provider.

    Example: tell me more about Carolina, or what is the rating of Arturo
    """

    print(name)
    profile = db.collection("profiles")

    docs = profile.where("displayName", "==", name).get()

    if len(docs) > 0:
        user_profile = docs[0].to_dict()
        print(user_profile)

        if user_profile["location"] is not None:
            user_profile["location"] = {
                "lat": docs[0].to_dict()["location"].latitude,
                "long": docs[0].to_dict()["location"].longitude,
            }

        return user_profile
    else:
        return None


@tool
def get_yomap_service_categories():
    """Get list of service categories from the database.
    Use this tool each time you need to know all the categories (services)
    on the database. Use this tool also if the user want to know about the
    services in the app.
    You can use this tool when the user is requesting for something and you can't
    find any service provider, in that case you can search for similar services.

    Example: the user request for plumber by you have handyman or plomero.
    """
    tags_ref = (
        db.collection("tags")
        .where("usedBy", ">=", 1)
        .order_by("usedBy")
        # .where(filter=FieldFilter("rating", ">=", 3))
    )
    docs = tags_ref.limit_to_last(100).get()

    tags = []
    for doc in docs:
        if "text" in doc.to_dict().keys():
            tags.append(doc.to_dict()["text"])
    return tags


# More utils
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_name = configuration.get("user_name", None)
            state = {**state, "user_info": user_name}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


llm = ChatOpenAI(
    openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0, streaming=True
)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for YoMap Servive Provider App. "
            " Use the provided tools to search for categories, service providers, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            " In some cases the user can ask questions where you need to run some tools many times and then agregate the information. "
            " Example: what is the mean rating of the spa service providers. In this case you need to get the spa service providers first, "
            " then get the info of each of them to get the rating and then compute the mean value of the rating. "
            " If the return of some functions/tools is empyt or Error: IndexError('list index out of range') please ignore it and use "
            " the other results to provide de answer."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.datetime.now())

tools = [
    get_profile_info_from_firebase,
    get_yomap_service_categories,
    get_service_provider_from_firebase,
]

assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
yomap_graph = builder.compile(checkpointer=memory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    yomap_graph,
    path="/bot",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
