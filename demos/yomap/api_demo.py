import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, Request, Response, HTTPException, Cookie
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from uuid import uuid4
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from typing import Annotated
from langgraph.graph.message import add_messages

from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    public_key="pk-lf-f185c086-8326-4eed-94d1-fa5a0cbe151f",
    secret_key="sk-lf-97193cea-e143-4c6f-8aac-f13eb49db28b",
    host="http://localhost:3000"
)

# 🔥 Inicializar Firestore
cred = credentials.Certificate("goblob-95e2a-6add9b68fd5d.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# 🔹 Colección en Firestore
SESSION_COLLECTION = "langgraph_api_sessions"

# 🚀 Iniciar FastAPI
app = FastAPI()

# 🤖 Crear el chatbot con memoria
llm = ChatOpenAI(
    openai_api_key="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", 
    model="gpt-4o", temperature=0, streaming=True
)

chat = llm

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 📌 Configurar LangGraph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")
executor = graph.compile()


# 🔹 Función para iniciar sesión en Firestore
@app.post("/start-session/")
def start_session(response: Response):
    session_id = str(uuid4())
    db.collection(SESSION_COLLECTION).document(session_id).set({"messages": []})
    
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return {"session_id": session_id, "message": "Session started"}

# 🔹 Guardar mensajes en Firestore
def save_to_firestore(session_id: str, message: str, response: str):
    session_ref = db.collection(SESSION_COLLECTION).document(session_id)
    session = session_ref.get()
    
    if not session.exists:
        session_ref.set({"messages": []})
    
    session_data = session.to_dict().get("messages", [])
    session_data.append({"role": "user", "content": message})
    session_data.append({"role": "ai", "content": response})
    # session_data.append({"user": message, "bot": response})
    session_ref.update({"messages": session_data})

# 🤖 Endpoint para conversar con el chatbot
@app.post("/chat/")
def chat_with_bot(request: Request, user_message: str, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")

    session_ref = db.collection(SESSION_COLLECTION).document(session_id)
    session = session_ref.get()

    if not session.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    # Obtener historial de conversación desde Firestore
    memory = session.to_dict().get("messages", [])

    # Procesar respuesta con LangGraph
    # response = executor.invoke({"messages": memory + [{"user": user_message}]})
    response = executor.invoke({"messages": memory + [{"role": "user", "content": user_message}]}, config={"callbacks": [langfuse_handler]} )
    print(response["messages"][-1].content)
    bot_reply = response["messages"][-1].content

    # Guardar en Firestore
    save_to_firestore(session_id, user_message, bot_reply)

    return {"session_id": session_id, "bot_reply": bot_reply}

# 🔹 Obtener historial de conversación
@app.get("/history/")
def get_chat_history(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")

    session_ref = db.collection(SESSION_COLLECTION).document(session_id)
    session = session_ref.get()

    if not session.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "messages": session.to_dict().get("messages", [])}
