from fastapi import FastAPI

from app.routes import caption, health, retrieve

app = FastAPI(title="PRISM ML Microservice", version="0.1.0")

app.include_router(health.router)
app.include_router(retrieve.router)
app.include_router(caption.router)

# TODO: mount /internal/reconstruct (structured decomposition endpoint)
