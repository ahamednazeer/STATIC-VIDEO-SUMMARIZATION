"""
Background Worker
Separate process that handles video processing jobs.
Prevents FastAPI from blocking on heavy computations.
"""

import asyncio
import logging
from pathlib import Path

from config import settings
from database.models import (
    init_database, get_pending_job, update_job_status,
    JobStatus, log_pipeline_event, get_video
)
from modules.pipeline import process_video


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("worker")


class BackgroundWorker:
    """
    Background worker that processes video summarization jobs.
    
    Architecture:
    - Polls database for pending jobs
    - Processes one job at a time (configurable)
    - Updates job status at each stage
    - Handles failures gracefully
    """
    
    def __init__(self):
        self.running = False
        self.current_job_id = None
    
    async def start(self):
        """Start the background worker loop."""
        logger.info("Background worker starting...")
        await init_database()
        
        self.running = True
        
        while self.running:
            try:
                await self._process_next_job()
            except Exception as e:
                logger.error(f"Worker error: {e}")
            
            # Poll interval
            await asyncio.sleep(settings.WORKER_POLL_INTERVAL)
    
    async def stop(self):
        """Stop the background worker."""
        logger.info("Background worker stopping...")
        self.running = False
    
    async def _process_next_job(self):
        """Check for and process the next pending job."""
        # Get next pending job
        job = await get_pending_job()
        
        if not job:
            return  # No pending jobs
        
        job_id = job['id']
        video_id = job['video_id']
        self.current_job_id = job_id
        
        logger.info(f"Processing job {job_id} for video {video_id}")
        
        try:
            # Get video info
            video = await get_video(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            video_path = video['original_path']
            
            # Mark as uploaded/starting
            await update_job_status(job_id, JobStatus.UPLOADED)
            await log_pipeline_event(job_id, "worker", f"Job picked up by worker")
            
            # Process the video
            await process_video(job_id, video_id, video_path)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            await update_job_status(job_id, JobStatus.FAILED, error=str(e))
        
        finally:
            self.current_job_id = None


async def run_worker():
    """Entry point for running the worker."""
    worker = BackgroundWorker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(run_worker())
