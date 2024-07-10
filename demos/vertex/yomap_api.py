# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Flask app that uses Vertex AI and Nominatim to get address coordinates.

This app takes an address as input and uses the Vertex AI Gemini model to
extract relevant location information. It then uses the Nominatim API to
retrieve the coordinates for the address.
"""

import json
import logging
import os

from typing import Any, List, Union

from flask import Flask, render_template, request
import requests
import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
)

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
from langchain_google_vertexai import ChatVertexAI

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account.
cred = credentials.Certificate("goblob-95e2a-0e1184d35308.json")

app = firebase_admin.initialize_app(cred)

db = firestore.client()

logger = logging.getLogger(__name__)

get_service_categories = FunctionDeclaration(
    name="get_service_categories",
    description="Get service categories from the database",
    parameters={
        "type": "object",
        "properties": {},
    },
)

get_service_provider = FunctionDeclaration(
    name="get_service_provider",
    description="Get service providers based on the tags",
    parameters={
        "type": "object",
        "properties": {
            "tag": {
                "type": "string",
                "description": "the category of the service the user is looking for",
            }
        },
    },
)

yomap_tool = Tool(
    function_declarations=[get_service_categories, get_service_provider],
)

tools = [get_service_categories, get_service_provider]


def get_service_categories_from_firebase():
    tags_ref = (
        db.collection("tags")
        # .where(filter=FieldFilter("active", "==", True))
        # .where(filter=FieldFilter("rating", ">=", 3))
    )
    docs = tags_ref.stream()

    tags = []
    for doc in docs:
        if "text" in doc.to_dict().keys():
            print(f'{doc.id} => {doc.to_dict()["text"]}')
            tags.append(doc.to_dict()["text"])
    return tags


def get_service_provider_from_firebase(tag: str):
    profile = db.collection("profiles")
    docs = profile.where("service.text", "==", tag["tag"]).get()
    return [doc.to_dict()["displayName"] for doc in docs]


function_handler = {
    "get_service_categories": get_service_categories_from_firebase,
    "get_service_provider": get_service_provider_from_firebase,
}

llm = ChatVertexAI(model_name="gemini-1.5-pro-001")

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

llm_with_tools = llm.bind(tools=tools)

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

agent_executor = AgentExecutor(agent=agent, verbose=True)

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
