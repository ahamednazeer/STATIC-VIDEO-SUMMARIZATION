/**
 * API client for Video Summarization backend
 */

import type {
    UploadResponse,
    JobStatusResponse,
    SummaryResponse,
    PipelineConfig,
    JobListItem
} from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiError extends Error {
    constructor(public status: number, message: string) {
        super(message);
        this.name = 'ApiError';
    }
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers: {
            ...options?.headers,
        },
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new ApiError(response.status, error.detail || response.statusText);
    }

    return response.json();
}

export const api = {
    /**
     * Check API health
     */
    health: () => fetchApi<{ status: string; service: string }>('/api/health'),

    /**
     * Get pipeline configuration
     */
    getConfig: () => fetchApi<PipelineConfig>('/api/config'),

    /**
     * Upload a video for processing
     */
    uploadVideo: async (file: File): Promise<UploadResponse> => {
        const formData = new FormData();
        formData.append('video', file);

        return fetchApi<UploadResponse>('/api/videos/upload', {
            method: 'POST',
            body: formData,
        });
    },

    /**
     * Get job status
     */
    getJobStatus: (jobId: number) =>
        fetchApi<JobStatusResponse>(`/api/jobs/${jobId}`),

    /**
     * Get summary for completed job
     */
    getSummary: (jobId: number) =>
        fetchApi<SummaryResponse>(`/api/jobs/${jobId}/summary`),

    /**
     * List all jobs
     */
    listJobs: () =>
        fetchApi<{ jobs: JobListItem[]; total: number }>('/api/jobs'),

    /**
     * Get keyframe image URL
     */
    getKeyframeUrl: (jobId: number, index: number) =>
        `${API_BASE}/api/jobs/${jobId}/keyframes/${index}`,

    /**
     * Get storyboard image URL
     */
    getStoryboardUrl: (jobId: number) =>
        `${API_BASE}/api/jobs/${jobId}/storyboard`,

    /**
     * Get download all keyframes URL (ZIP)
     */
    getDownloadAllUrl: (jobId: number) =>
        `${API_BASE}/api/jobs/${jobId}/download-all`,

    /**
     * Get original video URL for playback
     */
    getVideoUrl: (jobId: number) =>
        `${API_BASE}/api/jobs/${jobId}/video`,

    /**
     * Get storage file URL
     */
    getStorageUrl: (path: string) =>
        `${API_BASE}/storage/${path}`,
};

export { ApiError };
