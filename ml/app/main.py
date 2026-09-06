from fastapi import FastAPI

from app.routes import health

app = FastAPI(title="PRISM ML Microservice", version="0.1.0")

app.include_router(health.router)

# TODO: mount /internal/reconstruct (structured decomposition endpoint)
# TODO: mount /internal/retrieve (embedding + FAISS retrieval)
