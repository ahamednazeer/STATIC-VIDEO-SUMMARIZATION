"""
Database models and connection management for video summarization.
Implements explicit job state machine with 10 distinct states.
Uses aiosqlite for async SQLite operations.
"""

import aiosqlite
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from config import settings


def safe_int(value, default: int = 0) -> int:
    """Safely convert a value to int, handling bytes/None/corrupted data."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        try:
            # Try to decode as 8-byte little-endian int (SQLite 64-bit int)
            return int.from_bytes(value, 'little') if len(value) >= 4 else default
        except:
            return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


class JobStatus(str, Enum):
    """
    Explicit Job State Machine
    
    State Transitions:
    PENDING → UPLOADED → EXTRACTING_FRAMES → FILTERING_REDUNDANCY 
    → EXTRACTING_FEATURES → CLUSTERING → SELECTING_KEYFRAMES 
    → GENERATING_SUMMARY → COMPLETED
    
    Any state can transition to FAILED on error.
    """
    PENDING = "pending"                     # Job created, waiting to start
    UPLOADED = "uploaded"                   # Video file uploaded successfully
    EXTRACTING_FRAMES = "extracting_frames" # Decomposing video to frames
    FILTERING_REDUNDANCY = "filtering_redundancy"  # Removing similar frames
    EXTRACTING_FEATURES = "extracting_features"    # Computing histograms
    NORMALIZING = "normalizing"             # Normalizing feature vectors
    FINDING_OPTIMAL_K = "finding_optimal_k" # Elbow method
    CLUSTERING = "clustering"               # K-means clustering
    SELECTING_KEYFRAMES = "selecting_keyframes"    # Centroid selection
    GENERATING_SUMMARY = "generating_summary"      # Output generation
    COMPLETED = "completed"                 # Successfully finished
    FAILED = "failed"                       # Error occurred


# Progress percentages for each stage
STAGE_PROGRESS = {
    JobStatus.PENDING: 0,
    JobStatus.UPLOADED: 5,
    JobStatus.EXTRACTING_FRAMES: 15,
    JobStatus.FILTERING_REDUNDANCY: 30,
    JobStatus.EXTRACTING_FEATURES: 45,
    JobStatus.NORMALIZING: 55,
    JobStatus.FINDING_OPTIMAL_K: 65,
    JobStatus.CLUSTERING: 75,
    JobStatus.SELECTING_KEYFRAMES: 85,
    JobStatus.GENERATING_SUMMARY: 95,
    JobStatus.COMPLETED: 100,
    JobStatus.FAILED: 0,
}


@dataclass
class Video:
    """Video metadata record."""
    id: Optional[int]
    filename: str
    original_path: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int
    file_size: int
    created_at: datetime


@dataclass
class Job:
    """Processing job with state machine status."""
    id: Optional[int]
    video_id: int
    status: JobStatus
    progress: float
    current_stage: str
    stage_detail: Optional[str]  # e.g., "Processing frame 150/500"
    error_message: Optional[str]
    frames_extracted: int
    frames_filtered: int
    video_pacing: Optional[str] = None
    summary_confidence: Optional[str] = None
    stability_ratio: Optional[float] = None
    compression_ratio: Optional[float] = None
    redundancy_removed: Optional[float] = None
    temporal_coverage_score: Optional[float] = None
    summary_reason: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Keyframe:
    """Selected keyframe in summary."""
    id: Optional[int]
    job_id: int
    frame_index: int
    timestamp: float
    cluster_id: int
    distance_to_centroid: float
    output_path: str
    is_representative: bool


async def init_database():
    """Initialize database with required tables."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.executescript("""
            -- Videos table: stores video metadata
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                fps REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                duration REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                file_size INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Jobs table: explicit state machine for processing
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                progress REAL DEFAULT 0.0,
                current_stage TEXT DEFAULT 'pending',
                stage_detail TEXT,
                error_message TEXT,
                frames_extracted INTEGER DEFAULT 0,
                frames_filtered INTEGER DEFAULT 0,
                clusters_found INTEGER DEFAULT 0,
                -- Metadata fields
                video_pacing TEXT,
                summary_confidence TEXT,
                stability_ratio REAL,
                compression_ratio REAL,
                redundancy_removed REAL,
                temporal_coverage_score REAL,
                summary_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            );
            
            -- Keyframes table: selected summary frames
            CREATE TABLE IF NOT EXISTS keyframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                cluster_id INTEGER NOT NULL,
                distance_to_centroid REAL NOT NULL,
                output_path TEXT NOT NULL,
                is_representative BOOLEAN DEFAULT 1,
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );
            
            -- Pipeline logs for debugging
            CREATE TABLE IF NOT EXISTS pipeline_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                message TEXT NOT NULL,
                level TEXT DEFAULT 'info',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );
            
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON jobs(video_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_keyframes_job_id ON keyframes(job_id);
            CREATE INDEX IF NOT EXISTS idx_logs_job_id ON pipeline_logs(job_id);
        """)
        await db.commit()

        # Migration for existing databases: check for new columns
        cursor = await db.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in await cursor.fetchall()]

        new_columns = [
            ("video_pacing", "TEXT"),
            ("summary_confidence", "TEXT"),
            ("stability_ratio", "REAL"),
            ("compression_ratio", "REAL"),
            ("redundancy_removed", "REAL"),
            ("temporal_coverage_score", "REAL"),
            ("summary_reason", "TEXT")
        ]

        for col_name, col_type in new_columns:
            if col_name not in columns:
                await db.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}")

        await db.commit()


async def create_video(
    filename: str,
    original_path: str,
    fps: float,
    total_frames: int,
    duration: float,
    width: int,
    height: int,
    file_size: int = 0
) -> int:
    """Insert a new video record. Returns video ID."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO videos 
               (filename, original_path, fps, total_frames, duration, width, height, file_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, original_path, fps, total_frames, duration, width, height, file_size)
        )
        await db.commit()
        return cursor.lastrowid


async def create_job(video_id: int) -> int:
    """Create a new processing job. Returns job ID."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO jobs (video_id, status, current_stage) VALUES (?, ?, ?)",
            (video_id, JobStatus.PENDING.value, JobStatus.PENDING.value)
        )
        await db.commit()
        return cursor.lastrowid


async def update_job_status(
    job_id: int,
    status: JobStatus,
    stage_detail: str = None,
    error: str = None,
    **kwargs
):
    """
    Update job status with state machine tracking.
    
    Additional kwargs can include:
    - frames_extracted: int
    - frames_filtered: int  
    - clusters_found: int
    """
    progress = STAGE_PROGRESS.get(status, 0)
    
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        if status == JobStatus.COMPLETED:
            await db.execute(
                """UPDATE jobs SET 
                   status = ?, progress = 100.0, current_stage = ?,
                   stage_detail = ?, updated_at = CURRENT_TIMESTAMP,
                   completed_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (status.value, status.value, stage_detail, job_id)
            )
        elif status == JobStatus.FAILED:
            await db.execute(
                """UPDATE jobs SET 
                   status = ?, current_stage = ?, error_message = ?,
                   updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (status.value, status.value, error, job_id)
            )
        else:
            # Build dynamic update for additional fields
            updates = ["status = ?", "progress = ?", "current_stage = ?", 
                      "stage_detail = ?", "updated_at = CURRENT_TIMESTAMP"]
            values = [status.value, progress, status.value, stage_detail]
            
            # Handle first processing state
            if status == JobStatus.EXTRACTING_FRAMES:
                updates.append("started_at = CURRENT_TIMESTAMP")
            
            for key in ["frames_extracted", "frames_filtered", "clusters_found"]:
                if key in kwargs:
                    updates.append(f"{key} = ?")
                    values.append(kwargs[key])
            
            values.append(job_id)
            sql = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
            await db.execute(sql, values)
        
        await db.commit()


async def log_pipeline_event(job_id: int, stage: str, message: str, level: str = "info"):
    """Log a pipeline event for debugging."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.execute(
            "INSERT INTO pipeline_logs (job_id, stage, message, level) VALUES (?, ?, ?, ?)",
            (job_id, stage, message, level)
        )
        await db.commit()


async def get_job(job_id: int) -> Optional[dict]:
    """Get job by ID with all details."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_job_with_video(job_id: int) -> Optional[dict]:
    """Get job with associated video info."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT j.*, v.filename, v.duration, v.fps, v.width, v.height
            FROM jobs j
            JOIN videos v ON j.video_id = v.id
            WHERE j.id = ?
        """, (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_video(video_id: int) -> Optional[dict]:
    """Get video by ID."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_pending_job() -> Optional[dict]:
    """Get the next pending job for the worker."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1",
            (JobStatus.PENDING.value,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def save_keyframes(job_id: int, keyframes: List[dict]):
    """Save extracted keyframes."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.executemany(
            """INSERT INTO keyframes 
               (job_id, frame_index, timestamp, cluster_id, distance_to_centroid, output_path, is_representative)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(job_id, kf['frame_index'], kf['timestamp'], kf['cluster_id'],
              kf['distance_to_centroid'], kf['output_path'], kf.get('is_representative', True)) 
             for kf in keyframes]
        )
        await db.commit()


async def get_keyframes(job_id: int) -> List[dict]:
    """Get all keyframes for a job, ordered by timestamp."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM keyframes WHERE job_id = ? ORDER BY timestamp",
            (job_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_all_jobs() -> List[dict]:
    """Get all jobs with video info and aggregated keyframes for visual history."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Using ROW_NUMBER() to match sequential indices (keyframe_001.jpg etc.) 
        # and COALESCE to ensure we always have numbers for the UI.
        cursor = await db.execute("""
            SELECT j.*, v.filename, v.duration, v.total_frames as total_frames_original,
                   COALESCE(j.clusters_found, 0) as num_keyframes,
                   (
                       SELECT json_group_array(
                           json_object(
                               'index', k.seq_idx,
                               'frame_index', k.frame_index,
                               'timestamp', k.timestamp,
                               'timestamp_formatted', printf('%02d:%02d', CAST(k.timestamp / 60 AS INT), CAST(k.timestamp % 60 AS INT)),
                               'importance_score', 0.8,
                               'importance_confidence', 'MEDIUM'
                           )
                       )
                       FROM (
                           SELECT *, ROW_NUMBER() OVER (ORDER BY timestamp) as seq_idx 
                           FROM keyframes 
                           WHERE job_id = j.id
                       ) k
                   ) as keyframes_json
            FROM jobs j
            JOIN videos v ON j.video_id = v.id
            ORDER BY j.created_at DESC
        """)
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            # Robust property mapping and type safety using safe_int to handle bytes
            d['num_keyframes'] = safe_int(d.get('num_keyframes'), 0)
            d['total_frames_original'] = safe_int(d.get('total_frames_original'), 0)
            
            try:
                # Load keyframes and inject sequential index (1, 2, 3...) to match filesystem naming
                keyframes = json.loads(d.pop('keyframes_json')) if d.get('keyframes_json') else []
                for idx, kf in enumerate(keyframes):
                    kf['index'] = idx + 1
                d['keyframes'] = keyframes
            except:
                d['keyframes'] = []
            
            # Fallback for num_keyframes if count is missing but data is present
            if d['num_keyframes'] == 0 and d['keyframes']:
                d['num_keyframes'] = len(d['keyframes'])
                
            result.append(d)
        return result


async def get_pipeline_logs(job_id: int) -> List[dict]:
    """Get all logs for a job."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM pipeline_logs WHERE job_id = ? ORDER BY created_at",
            (job_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
