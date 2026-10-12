from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from src.self_healing_rag import run_self_healing_rag


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Self-Healing RAG API",
    description="API for the Self-Healing Government Policy RAG System",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Self-Healing RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        print("\n")
        print("=" * 70)
        print("API REQUEST")
        print("=" * 70)

        print(f"Question: {question}")

        # Run the existing Self-Healing RAG
        result = run_self_healing_rag(
            question,
            verbose=True
        )

        # If the RAG doesn't return anything
        if result is None:

            return {
                "success": True,
                "question": question,
                "answer": "RAG executed successfully, but no result was returned."
            }

        # If RAG returns a dictionary
        if isinstance(result, dict):

            return {
                "success": True,
                "question": question,
                "result": make_json_safe(result)
            }

        # If RAG returns a string or another object
        return {
            "success": True,
            "question": question,
            "answer": str(result)
        }

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("API ERROR")
        print("=" * 70)

        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# MAKE RESULT JSON SAFE
# ============================================================

def make_json_safe(value: Any):

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool)
    ):
        return value

    if isinstance(value, dict):

        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        (list, tuple)
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    return str(value)