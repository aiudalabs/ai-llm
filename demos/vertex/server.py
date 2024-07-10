#!/usr/bin/env python
"""Example LangChain server exposes and agent that has conversation history.

In this example, the history is stored entirely on the client's side.

Please see other examples in LangServe on how to use RunnableWithHistory to
store history on the server side.

Relevant LangChain documentation:

* Creating a custom agent: https://python.langchain.com/docs/modules/agents/how_to/custom_agent
* Streaming with agents: https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events
* General streaming documentation: https://python.langchain.com/docs/expression_language/streaming
* Message History: https://python.langchain.com/docs/expression_language/how_to/message_history

**ATTENTION**
1. To support streaming individual tokens you will need to use the astream events
   endpoint rather than the streaming endpoint.
2. This example does not truncate message history, so it will crash if you
   send too many messages (exceed token length).
3. The playground at the moment does not render agent output well! If you want to
   use the playground you need to customize it's output server side using astream
   events by wrapping it within another runnable.
4. See the client notebook it has an example of how to use stream_events client side!
"""  # noqa: E501
from typing import Any, List, Union

from fastapi import FastAPI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.utils.function_calling import format_tool_to_openai_tool
from langchain_openai import ChatOpenAI

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Use a service account.
cred = credentials.Certificate("goblob-95e2a-0e1184d35308.json")

app = firebase_admin.initialize_app(cred)

db = firestore.client()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are service provider assistant, but you don't know anything about services or categories."
            "So, each time the user ask you about the service or categories use a proper tool to answer correctly."
            "Talk with the user as normal. "
            "If they ask you about any category or service, use a tool",
        ),
        # Please note the ordering of the fields in the prompt!
        # The correct ordering is:
        # 1. history - the past messages between the user and the agent
        # 2. user - the user's current input
        # 3. agent_scratchpad - the agent's working space for thinking and
        #    invoking tools to respond to the user's input.
        # If you change the ordering, the agent will not work correctly since
        # the messages will be shown to the underlying LLM in the wrong order.
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


@tool
def service_categories() -> int:
    """use this tool when a user ask for a categories
    It will return the user profile in JSON format."""

    users_ref = (
        db.collection("tags")
        # .where(filter=FieldFilter("active", "==", True))
        # .where(filter=FieldFilter("rating", ">=", 3))
    )
    docs = users_ref.stream()

    providers = []
    for doc in docs:
        if "text" in doc.to_dict().keys():
            print(f'{doc.id} => {doc.to_dict()["text"]}')
            providers.append(doc.to_dict()["text"])
    return providers


@tool
def get_profiles(query_str: str):
    """Use this tool to answer any query related to a profile or a service provider"""
    return query(query_str)


def query(query_str):
    profile = db.collection("profiles")
    # Esta función necesita analizar la consulta y decidir qué hacer
    try:
        # Por ejemplo, buscar documentos por un campo específico
        if "age" in query_str:
            age = int(query_str.split("age:")[1])
            print("Age query with age = " + str(age))
            docs = profile.where("age", "==", age).get()
            for doc in docs:
                print(doc.to_dict()["displayName"])
            return [doc.to_dict() for doc in docs]
        elif "tag" in query_str:
            tag = query_str.split("tag:")[1]
            print("Tags query with tag = " + str(tag))
            docs = profile.where("service.text", "==", tag).get()
            return [doc.to_dict() for doc in docs]
        else:
            return ["Consulta no reconocida"]
    except Exception as e:
        return [str(e)]


# We need to set streaming=True on the LLM to support streaming individual tokens.
# Tokens will be available when using the stream_log / stream events endpoints,
# but not when using the stream endpoint since the stream implementation for agent
# streams action observation pairs not individual tokens.
# See the client notebook that shows how to use the stream events endpoint.
llm = ChatOpenAI(
    openai_api_key="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",
    model="gpt-3.5-turbo",
    temperature=0,
    streaming=True,
)

tools = [service_categories, get_profiles]


llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])

# ATTENTION: For production use case, it's a good idea to trim the prompt to avoid
#            exceeding the context window length used by the model.
#
# To fix that simply adjust the chain to trim the prompt in whatever way
# is appropriate for your use case.
# For example, you may want to keep the system message and the last 10 messages.
# Or you may want to trim based on the number of tokens.
# Or you may want to also summarize the messages to keep information about things
# that were learned about the user.
#
# def prompt_trimmer(messages: List[Union[HumanMessage, AIMessage, FunctionMessage]]):
#     '''Trims the prompt to a reasonable length.'''
#     # Keep in mind that when trimming you may want to keep the system message!
#     return messages[-10:] # Keep last 10 messages.

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    # | prompt_trimmer # See comment above.
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)


# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str
    # The field extra defines a chat widget.
    # Please see documentation about widgets in the main README.
    # The widget is used in the playground.
    # Keep in mind that playground support for agents is not great at the moment.
    # To get a better experience, you'll need to customize the streaming output
    # for now.
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )


class Output(BaseModel):
    output: Any


# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
# /stream_events
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
