from fastapi import FastAPI

from app.routes import health

app = FastAPI(title="PRISM ML Microservice", version="0.1.0")

app.include_router(health.router)

# TODO(Task 0.2): mount /internal/reconstruct once shared API contract (Pydantic models) is finalized
# TODO(Task 2.4): mount /internal/retrieve (embedding + FAISS retrieval)
