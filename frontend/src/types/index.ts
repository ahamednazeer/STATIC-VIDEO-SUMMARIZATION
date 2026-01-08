/**
 * TypeScript type definitions for the Video Summarization API
 */

export type JobStatus =
  | 'pending'
  | 'uploaded'
  | 'extracting_frames'
  | 'filtering_redundancy'
  | 'extracting_features'
  | 'normalizing'
  | 'finding_optimal_k'
  | 'clustering'
  | 'selecting_keyframes'
  | 'generating_summary'
  | 'completed'
  | 'failed';

export interface UploadResponse {
  success: boolean;
  job_id: number;
  video_id: number;
  message: string;
}

export interface JobStatusResponse {
  job_id: number;
  status: JobStatus;
  progress: number;
  current_stage: string;
  stage_detail: string | null;
  error_message: string | null;
  frames_extracted: number;
  frames_filtered: number;
  clusters_found: number;
  video_filename: string | null;
  video_duration: number | null;
}

export interface KeyframeInfo {
  index: number;
  frame_id: number;
  timestamp: number;
  timestamp_formatted: string;
  cluster_id: number;
  distance_to_centroid: number;
  path: string;
  context?: string;
  domain?: string;
  domain_confidence?: number;
  deep_analysis?: {
    objects: Array<{ label: string; score: number }>;
    actions: Array<{ label: string; score: number }>;
    environment: Array<{ label: string; score: number }>;
    attributes: Array<{ label: string; score: number }>;
  };
  // Importance scoring fields (optional for backward compatibility)
  importance_score?: number;
  importance_confidence?: 'HIGH' | 'MEDIUM' | 'LOW';
  importance_factors?: {
    representativeness?: number;
    dominance?: number;
    visual_richness?: number;
    temporal_coverage?: number;
    visual_novelty?: number;
    motion_context?: number;
  };
}

export interface SummaryResponse {
  job_id: number;
  video_filename: string;
  video_duration: number;
  total_frames_original: number;
  frames_after_filtering: number;
  num_keyframes: number;
  keyframes: KeyframeInfo[];
  grid_available: boolean;
  // Importance metadata (optional)
  video_pacing?: 'Fast' | 'Moderate' | 'Slow' | 'fast' | 'normal' | 'slow';
  summary_confidence?: 'HIGH' | 'MEDIUM' | 'LOW';
  stability_ratio?: number;
  // NEW: Advanced metrics
  compression_ratio?: number;
  redundancy_removed?: number;
  temporal_coverage_score?: number;
  summary_reason?: string;
}

export interface PipelineConfig {
  max_video_duration_seconds: number;
  max_video_size_mb: number;
  supported_formats: string[];
  redundancy_threshold: number;
  color_space: string;
  min_clusters: number;
  max_clusters: number;
  output_format: string;
}

export interface JobListItem {
  id: number;
  video_id: number;
  status: JobStatus;
  progress: number;
  current_stage: string;
  filename: string;
  duration: number;
  created_at: string;
  completed_at: string | null;
  // Summary metrics (added for detailed history)
  video_pacing?: string;
  summary_confidence?: 'HIGH' | 'MEDIUM' | 'LOW';
  stability_ratio?: number;
  compression_ratio?: number;
  redundancy_removed?: number;
  temporal_coverage_score?: number;
  num_keyframes?: number;
  total_frames_original?: number;
  summary_reason?: string;
  keyframes?: KeyframeInfo[];
}

// Stage information for progress display
export const STAGE_INFO: Record<JobStatus, { label: string; icon: string; order: number }> = {
  pending: { label: 'Pending', icon: 'clock', order: 0 },
  uploaded: { label: 'Uploaded', icon: 'upload', order: 1 },
  extracting_frames: { label: 'Extracting Frames', icon: 'film', order: 2 },
  filtering_redundancy: { label: 'Filtering Redundancy', icon: 'filter', order: 3 },
  extracting_features: { label: 'Extracting Features', icon: 'cpu', order: 4 },
  normalizing: { label: 'Normalizing', icon: 'sliders', order: 5 },
  finding_optimal_k: { label: 'Finding Optimal K', icon: 'target', order: 6 },
  clustering: { label: 'Clustering', icon: 'box', order: 7 },
  selecting_keyframes: { label: 'Selecting Keyframes', icon: 'image', order: 8 },
  generating_summary: { label: 'Generating Summary', icon: 'layout', order: 9 },
  completed: { label: 'Completed', icon: 'check-circle', order: 10 },
  failed: { label: 'Failed', icon: 'x-circle', order: -1 },
};
