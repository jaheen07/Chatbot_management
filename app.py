from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from datetime import datetime
import logging
import threading
from chatbot_groq import RAGChatbot

"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"

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
    allow_origins=[
        "*",  # Allow all for development
        "https://*.ngrok-free.app",  # Ngrok domains
        "https://*.ngrok.io",  # Old ngrok domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
base_chatbot = None
user_sessions = {}
session_lock = threading.Lock()

# Configuration
MAX_SESSIONS = 100
SESSION_TIMEOUT = 3600  # 1 hour


class UserSession:
    """Isolated session for each user"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.chat_history = []
        self.conversation_context = {
            'current_entities': [],
            'entity_attributes': {},
            'numerical_context': {}
        }
        self.last_activity = datetime.now()

    def update_activity(self):
        self.last_activity = datetime.now()


def cleanup_old_sessions():
    """Remove inactive sessions"""
    with session_lock:
        current_time = datetime.now()
        to_remove = []

        for user_id, session in user_sessions.items():
            time_diff = (current_time - session.last_activity).total_seconds()
            if time_diff > SESSION_TIMEOUT:
                to_remove.append(user_id)

        for user_id in to_remove:
            del user_sessions[user_id]
            logger.info(f"Cleaned up session for user: {user_id}")


def get_or_create_session(user_id: str) -> UserSession:
    """Get existing session or create new one"""
    with session_lock:
        if len(user_sessions) > MAX_SESSIONS:
            cleanup_old_sessions()

        if user_id not in user_sessions:
            user_sessions[user_id] = UserSession(user_id)
            logger.info(f"Created new session for user: {user_id}")

        session = user_sessions[user_id]
        session.update_activity()
        return session


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
    current_context: Optional[str]
    history: List[Dict]


@app.on_event("startup")
async def startup_event():
    global base_chatbot

    logger.info("=== Starting RAG Chatbot Initialization ===")

    try:
        PDF_PATH = os.getenv("PDF_PATH", "./data/Policies.pdf")
        CSV_PATH = os.getenv("CSV_PATH", "./data/employee_table.csv")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
        "active_sessions": len(user_sessions),
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

    stats = base_chatbot.get_stats()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chatbot_ready": True,
        "active_sessions": len(user_sessions),
        "total_chunks": stats['total_chunks'],
        "model": base_chatbot.model_name
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

        # Get user session
        session = get_or_create_session(request.user_id)

        # Use base chatbot's methods but with session-specific context
        resolved_question = request.question

        # Resolve pronouns using session context
        if base_chatbot._detect_pronouns(request.question):
            if session.conversation_context['current_entities']:
                recent_entity = session.conversation_context['current_entities'][-1]

                pronoun_map = {
                    r'\b(he|she)\b': recent_entity,
                    r'\b(him|her)\b': recent_entity,
                    r'\b(his|her|their)\b': f"{recent_entity}'s",
                    r'\bthey\b': recent_entity,
                    r'\bthem\b': recent_entity
                }

                import re
                for pattern, replacement in pronoun_map.items():
                    resolved_question = re.sub(pattern, replacement, resolved_question, flags=re.IGNORECASE)

                logger.info(f"[Pronoun Resolution] {recent_entity}")

        # Retrieve relevant chunks from base chatbot
        retrieved_data = base_chatbot._retrieve(resolved_question, k=15)

        # Search session-specific chat history
        relevant_past_chats = []
        if len(session.chat_history) > 0:
            # Simple keyword matching in session history
            query_lower = resolved_question.lower()
            for entry in session.chat_history[-10:]:
                if any(word in entry['question'].lower() for word in query_lower.split()[:3]):
                    relevant_past_chats.append({'chat': entry})

        # Perform calculations
        context_text = "\n".join([chunk for chunk, _, _ in retrieved_data])
        calculation_result = base_chatbot._perform_calculations(resolved_question, context_text)

        # Build prompt using base chatbot method
        prompt = base_chatbot._build_prompt(
            resolved_question,
            retrieved_data,
            relevant_past_chats,
            calculation_result
        )

        # Call Groq API
        import requests
        payload = {
            "model": base_chatbot.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.2
        }

        response = requests.post(
            base_chatbot.api_url,
            headers=base_chatbot.headers,
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        result = response.json()

        # Extract answer
        answer = result["choices"][0]["message"]["content"]

        # Update session context
        entities = base_chatbot._extract_entities_from_text(request.question + " " + answer)
        for entity in entities:
            if entity not in session.conversation_context['current_entities']:
                session.conversation_context['current_entities'].append(entity)
                # Keep only last 10 entities
                if len(session.conversation_context['current_entities']) > 10:
                    session.conversation_context['current_entities'].pop(0)

        # Store in session history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': request.question,
            'resolved_question': resolved_question,
            'answer': answer,
            'used_past_context': len(relevant_past_chats) > 0
        }
        session.chat_history.append(chat_entry)

        response_data = ChatResponse(
            question=request.question,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            user_id=request.user_id,
            session_info={
                'total_messages': len(session.chat_history),
                'current_entities': session.conversation_context['current_entities'][-3:] if
                session.conversation_context['current_entities'] else []
            }
        )

        logger.info(f"User {request.user_id}: Question processed successfully")
        return response_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Groq API Error for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

    except Exception as e:
        logger.error(f"Error for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/reset/{user_id}")
async def reset_chat(user_id: str):
    """Reset chat history for specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    with session_lock:
        if user_id in user_sessions:
            del user_sessions[user_id]
            logger.info(f"Reset session for user: {user_id}")
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
    session = get_or_create_session(user_id)

    history = session.chat_history
    if limit and limit > 0:
        history = history[-limit:]

    current_entity = None
    if session.conversation_context['current_entities']:
        current_entity = session.conversation_context['current_entities'][-1]

    return HistoryResponse(
        user_id=user_id,
        total_conversations=len(session.chat_history),
        current_context=current_entity,
        history=history
    )


@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    with session_lock:
        return {
            "total_sessions": len(user_sessions),
            "max_sessions": MAX_SESSIONS,
            "session_timeout_seconds": SESSION_TIMEOUT,
            "sessions": [
                {
                    "user_id": user_id,
                    "messages": len(session.chat_history),
                    "last_activity": session.last_activity.isoformat(),
                    "current_entities": session.conversation_context['current_entities'][-3:] if
                    session.conversation_context['current_entities'] else []
                }
                for user_id, session in user_sessions.items()
            ]
        }


@app.post("/api/cleanup")
async def manual_cleanup():
    """Manually trigger session cleanup"""
    cleanup_old_sessions()
    return {
        "message": "Cleanup completed",
        "active_sessions": len(user_sessions)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
