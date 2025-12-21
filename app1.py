from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from datetime import datetime
import logging
import threading
from new_chatbot import RAGChatbot, ConversationMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API - Multi-User",
    description="HR Assistant Chatbot with Per-User Session Management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
base_chatbot = None
user_bots = {}
session_lock = threading.Lock()

# Configuration
MAX_SESSIONS = 100
SESSION_TIMEOUT = 3600  # 1 hour


class UserBotInstance:
    """Isolated chatbot instance per user"""

    def __init__(self, base_bot: RAGChatbot, user_id: str):
        self.user_id = user_id
        self.bot = base_bot
        self.memory = ConversationMemory()
        self.last_activity = datetime.now()

    def update_activity(self):
        self.last_activity = datetime.now()


def cleanup_old_sessions():
    """Remove inactive sessions"""
    with session_lock:
        current_time = datetime.now()
        to_remove = []

        for user_id, instance in user_bots.items():
            time_diff = (current_time - instance.last_activity).total_seconds()
            if time_diff > SESSION_TIMEOUT:
                to_remove.append(user_id)

        for user_id in to_remove:
            del user_bots[user_id]
            logger.info(f"Cleaned up session for user: {user_id}")


def get_or_create_bot(user_id: str) -> UserBotInstance:
    """Get existing bot instance or create new one"""
    with session_lock:
        if len(user_bots) > MAX_SESSIONS:
            cleanup_old_sessions()

        if user_id not in user_bots:
            if base_chatbot is None:
                raise HTTPException(status_code=503, detail="Base chatbot not initialized")
            user_bots[user_id] = UserBotInstance(base_chatbot, user_id)
            logger.info(f"Created new bot instance for user: {user_id}")

        instance = user_bots[user_id]
        instance.update_activity()
        return instance


# Pydantic models
class ChatRequest(BaseModel):
    question: str
    user_id: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    timestamp: str
    user_id: str
    session_info: Dict


class HistoryResponse(BaseModel):
    user_id: str
    total_conversations: int
    current_entities: List[str]
    history: List[Dict]


@app.on_event("startup")
async def startup_event():
    global base_chatbot

    logger.info("=== Starting RAG Chatbot Initialization ===")

    try:
        PDF_PATH = os.getenv("PDF_PATH", "./data/Policies.pdf")
        CSV_PATH = os.getenv("CSV_PATH", "./data/employee_table.csv")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY", "a")

        logger.info(f"PDF Path: {PDF_PATH}")
        logger.info(f"CSV Path: {CSV_PATH}")
        logger.info(f"PDF exists: {os.path.exists(PDF_PATH)}")
        logger.info(f"CSV exists: {os.path.exists(CSV_PATH)}")

        if not os.path.exists(PDF_PATH):
            raise ValueError(f"PDF file not found at {PDF_PATH}")

        if not os.path.exists(CSV_PATH):
            raise ValueError(f"CSV file not found at {CSV_PATH}")

        base_chatbot = RAGChatbot(PDF_PATH, CSV_PATH, GROQ_API_KEY)
        logger.info("=== Base chatbot initialized successfully! ===")

    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise


@app.get("/")
async def root():
    return {
        "service": "RAG Chatbot API",
        "version": "2.0.0",
        "status": "healthy",
        "active_sessions": len(user_bots),
        "chatbot_loaded": base_chatbot is not None,
        "endpoints": {
            "docs": "/docs",
            "chat": "POST /api/chat",
            "history": "GET /api/history/{user_id}",
            "reset": "POST /api/reset/{user_id}",
            "sessions": "GET /api/sessions",
            "health": "GET /api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    if base_chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chatbot_ready": True,
        "active_sessions": len(user_bots),
        "total_chunks": len(base_chatbot.docs.chunks),
        "model": "llama-3.3-70b-versatile"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question to the chatbot with user session isolation"""
    if base_chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        logger.info(f"User {request.user_id}: {request.question[:50]}...")

        # Get user-specific bot instance
        user_instance = get_or_create_bot(request.user_id)

        # Create temporary bot with user's memory
        temp_bot = RAGChatbot.__new__(RAGChatbot)
        temp_bot.api_key = base_chatbot.api_key
        temp_bot.api_url = base_chatbot.api_url
        temp_bot.embedder = base_chatbot.embedder
        temp_bot.docs = base_chatbot.docs
        temp_bot.index = base_chatbot.index
        temp_bot.memory = user_instance.memory
        temp_bot.calculator = base_chatbot.calculator
        temp_bot._retrieve = base_chatbot._retrieve.__get__(temp_bot, RAGChatbot)
        temp_bot._build_prompt = base_chatbot._build_prompt.__get__(temp_bot, RAGChatbot)
        temp_bot._call_llm = base_chatbot._call_llm.__get__(temp_bot, RAGChatbot)

        # Process question
        answer = temp_bot.ask(request.question)

        # Update user instance memory
        user_instance.memory = temp_bot.memory

        response_data = ChatResponse(
            question=request.question,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            user_id=request.user_id,
            session_info={
                'total_messages': len(user_instance.memory.messages),
                'current_entities': list(user_instance.memory.entities.values())[
                    -3:] if user_instance.memory.entities else []
            }
        )

        logger.info(f"User {request.user_id}: Question processed successfully")
        return response_data

    except Exception as e:
        logger.error(f"Error for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/reset/{user_id}")
async def reset_chat(user_id: str):
    """Reset chat history for specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    with session_lock:
        if user_id in user_bots:
            user_bots[user_id].memory = ConversationMemory()
            logger.info(f"Reset memory for user: {user_id}")
            return {
                "message": f"Chat history reset for user {user_id}",
                "status": "success"
            }
        else:
            return {
                "message": f"No session found for user {user_id}",
                "status": "success"
            }


@app.get("/api/history/{user_id}", response_model=HistoryResponse)
async def get_history(user_id: str, limit: Optional[int] = None):
    """Get chat history for specific user"""
    user_instance = get_or_create_bot(user_id)

    messages = list(user_instance.memory.messages)
    if limit and limit > 0:
        messages = messages[-limit:]

    history = []
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            history.append({
                'question': messages[i].content,
                'answer': messages[i + 1].content
            })

    return HistoryResponse(
        user_id=user_id,
        total_conversations=len(history),
        current_entities=list(user_instance.memory.entities.values()),
        history=history
    )


@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    with session_lock:
        return {
            "total_sessions": len(user_bots),
            "max_sessions": MAX_SESSIONS,
            "session_timeout_seconds": SESSION_TIMEOUT,
            "sessions": [
                {
                    "user_id": user_id,
                    "messages": len(instance.memory.messages),
                    "last_activity": instance.last_activity.isoformat(),
                    "current_entities": list(instance.memory.entities.values())[-3:] if instance.memory.entities else []
                }
                for user_id, instance in user_bots.items()
            ]
        }


@app.post("/api/cleanup")
async def manual_cleanup():
    """Manually trigger session cleanup"""
    cleanup_old_sessions()
    return {
        "message": "Cleanup completed",
        "active_sessions": len(user_bots)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)