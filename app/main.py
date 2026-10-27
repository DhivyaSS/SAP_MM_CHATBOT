import logging
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager: initializing vector store on startup.
    """
    logger.info("Initializing SAP MM RAG Assistant application...")
    if getattr(settings, "VECTOR_STORE_TYPE", "chroma").lower() == "pgvector":
        try:
            init_db()
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL on startup: {e}")
    else:
        logger.info(f"Using local ChromaDB vector store at '{settings.CHROMA_PERSIST_DIRECTORY}'. (No PostgreSQL/Docker required).")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Grounded RAG Intelligent Knowledge Assistant for SAP Materials Management (MM)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for local Streamlit & frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "title": settings.APP_NAME,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
