"""
FastAPI Application Entry Point
Main server with CORS, background worker integration, and static file serving.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database.models import init_database
from api.routes import router
from worker import BackgroundWorker


# Global worker instance
worker = BackgroundWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes database and starts background worker on startup.
    Gracefully stops worker on shutdown.
    """
    # Startup
    print("Starting Video Summarization API...")
    
    # Initialize database
    await init_database()
    print("Database initialized")
    
    # Start background worker
    worker_task = asyncio.create_task(worker.start())
    print("Background worker started")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    print("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Static Video Summarization API",
    description="""
    API for converting long videos into meaningful static keyframe summaries.
    
    ## Features
    - Video upload and processing
    - Real-time job status tracking
    - Keyframe extraction using K-means clustering
    - Storyboard grid generation
    
    ## Pipeline Stages
    1. Frame Extraction
    2. Redundancy Filtering
    3. Feature Extraction (HSV Histograms)
    4. Normalization
    5. Elbow Method (Optimal K)
    6. K-Means Clustering
    7. Representative Selection
    8. Summary Generation
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static files for serving keyframe images
storage_path = settings.STORAGE_DIR
if storage_path.exists():
    app.mount(
        "/storage",
        StaticFiles(directory=str(storage_path)),
        name="storage"
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Static Video Summarization API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
