from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import build_graph
from app.routers import explain, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    # In-memory session store keyed by session_id (UUID string).
    # Replace with Redis for production deployments.
    app.state.sessions = {}
    # Pre-compile the LangGraph agent once at startup for efficiency.
    app.state.graph = build_graph()
    yield
    app.state.sessions.clear()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Anomaly Explainer Agent",
        description="Explains anomaly detection model predictions using SHAP, LIME, and PDP.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(upload.router)
    app.include_router(explain.router)

    return app


app = create_app()
