#!/usr/bin/env python3
"""
Run script for the Video Summarization Backend.
Starts both the FastAPI server and background worker.
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    from config import settings
    
    print("=" * 60)
    print("Static Video Summarization System")
    print("=" * 60)
    print(f"API Server: http://localhost:{settings.API_PORT}")
    print(f"API Docs:   http://localhost:{settings.API_PORT}/docs")
    print(f"Database:   {settings.DATABASE_PATH}")
    print(f"Storage:    {settings.STORAGE_DIR}")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        reload_dirs=["./"]
    )
