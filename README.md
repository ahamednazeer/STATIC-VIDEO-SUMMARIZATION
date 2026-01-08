# Static Video Summarization System

A complete system for converting long videos into meaningful static image summaries using K-means clustering and histogram-based feature extraction.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 16)                    │
│  React 19 │ TypeScript │ Tailwind CSS 4 │ Lucide & Phosphor    │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────────┐
│                     Backend (Python/FastAPI)                     │
│  ┌──────────────┐    ┌────────────────┐    ┌────────────────┐  │
│  │   API Layer  │───▶│ Background     │───▶│   SQLite DB    │  │
│  │   (Routes)   │    │ Worker         │    │   (Jobs)       │  │
│  └──────────────┘    └───────┬────────┘    └────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │                    Processing Pipeline                      │  │
│  │  Video → Frames → Filter → Features → K-Means → Summary  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1. Video Ingestion | Extract metadata (FPS, resolution, duration) |
| 2. Frame Extraction | Decompose video into discrete frames |
| 3. Redundancy Filter | Remove similar consecutive frames |
| 4. Feature Extraction | Compute HSV histograms (768-dim vectors) |
| 5. Normalization | L2 normalize features for clustering |
| 6. Elbow Method | Find optimal number of clusters |
| 7. K-Means Clustering | Group frames by visual similarity |
| 8. Representative Selection | Choose frame closest to each centroid |
| 9. Summary Generation | Save keyframes and create storyboard |

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server (includes background worker)
python main.py
```

Backend will be available at: `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET | Pipeline configuration |
| `/api/videos/upload` | POST | Upload video for processing |
| `/api/jobs/{id}` | GET | Get job status |
| `/api/jobs/{id}/summary` | GET | Get summary (when completed) |
| `/api/jobs/{id}/keyframes/{n}` | GET | Get keyframe image |
| `/api/jobs/{id}/storyboard` | GET | Get storyboard grid |

## Configuration

Edit `backend/config.py` to adjust pipeline parameters:

```python
# Clustering
MAX_CLUSTERS = 15
MIN_CLUSTERS = 2

# Redundancy filtering
REDUNDANCY_THRESHOLD = 0.95

# Feature extraction
COLOR_SPACE = "hsv"  # or "rgb"
HISTOGRAM_BINS = 256
```

## Output

For each processed video, the system generates:

- **Individual keyframes**: High-quality images of selected frames
- **Storyboard grid**: Combined visualization of all keyframes
- **Metadata JSON**: Frame indices, timestamps, cluster assignments

## Tech Stack

### Backend
- Python 3.10+
- FastAPI
- OpenCV
- scikit-learn
- SQLite (aiosqlite)

### Frontend
- Next.js 16 (App Router)
- React 19
- TypeScript
- Tailwind CSS 4
- Lucide React & Phosphor Icons

## Project Structure

```
├── backend/
│   ├── api/
│   │   └── routes.py         # REST endpoints
│   ├── database/
│   │   └── models.py         # SQLite models & queries
│   ├── modules/
│   │   ├── video_ingestion.py
│   │   ├── frame_extractor.py
│   │   ├── redundancy_filter.py
│   │   ├── feature_extractor.py
│   │   ├── clustering.py
│   │   ├── summary_generator.py
│   │   └── pipeline.py       # Orchestrator
│   ├── config.py             # Settings
│   ├── main.py               # FastAPI app
│   ├── worker.py             # Background processor
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx      # Main page
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── VideoUpload.tsx
│   │   │   ├── ProcessingProgress.tsx
│   │   │   └── SummaryGallery.tsx
│   │   ├── types/
│   │   │   └── index.ts
│   │   └── utils/
│   │       └── api.ts
│   └── package.json
│
└── README.md
```

## License

MIT
