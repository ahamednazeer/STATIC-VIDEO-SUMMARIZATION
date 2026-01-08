'use client';

import { useState, useEffect, useCallback } from 'react';
import { Sparkles, History, RefreshCw, Upload, Film, Cpu, LayoutGrid } from 'lucide-react';
import VideoUpload from '@/components/VideoUpload';
import ProcessingProgress from '@/components/ProcessingProgress';
import SummaryGallery from '@/components/SummaryGallery';
import HistoryModal from '@/components/HistoryModal';
import { api } from '@/utils/api';
import { JobStatusResponse, SummaryResponse, PipelineConfig } from '@/types';

type AppState = 'upload' | 'processing' | 'completed' | 'error';

export default function Home() {
  const [state, setState] = useState<AppState>('upload');
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState<number | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [config, setConfig] = useState<PipelineConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    api.getConfig().then(setConfig).catch(console.error);
  }, []);

  useEffect(() => {
    if (!jobId || state !== 'processing') return;
    const poll = setInterval(async () => {
      try {
        const status = await api.getJobStatus(jobId);
        setJobStatus(status);
        if (status.status === 'completed') {
          clearInterval(poll);
          setSummary(await api.getSummary(jobId));
          setState('completed');
        } else if (status.status === 'failed') {
          clearInterval(poll);
          setError(status.error_message || 'Processing failed');
          setState('error');
        }
      } catch (err) { console.error(err); }
    }, 1000);
    return () => clearInterval(poll);
  }, [jobId, state]);

  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);
    try {
      const result = await api.uploadVideo(file);
      setJobId(result.job_id);
      setState('processing');
      setJobStatus(await api.getJobStatus(result.job_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setState('error');
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handleReset = useCallback(() => {
    setState('upload');
    setJobId(null);
    setJobStatus(null);
    setSummary(null);
    setError(null);
  }, []);

  const handleSelectFromHistory = useCallback(async (selectedJobId: number) => {
    setShowHistory(false);
    try {
      const summaryData = await api.getSummary(selectedJobId);
      setSummary(summaryData);
      setJobId(selectedJobId);
      setState('completed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load job');
      setState('error');
    }
  }, []);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', padding: '32px 20px' }}>
      {/* Header */}
      <header style={{ maxWidth: '1200px', margin: '0 auto 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '46px', height: '46px', borderRadius: '12px',
            background: 'linear-gradient(135deg, #6366f1, #a855f7)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 6px 24px rgba(99, 102, 241, 0.3)'
          }}>
            <Sparkles size={24} color="white" />
          </div>
          <div>
            <h1 className="gradient-text" style={{ fontSize: '22px', fontWeight: 700, margin: 0 }}>
              Video Summarizer
            </h1>
            <p style={{ color: 'var(--muted)', fontSize: '13px', margin: '2px 0 0' }}>
              Extract keyframes using K-means clustering
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {state !== 'upload' && (
            <button onClick={handleReset} className="btn-secondary">
              <RefreshCw size={14} /> New Video
            </button>
          )}
          <button onClick={() => setShowHistory(true)} className="btn-secondary">
            <History size={14} /> History
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {state === 'upload' && (
          <>
            {/* Hero Section */}
            <div style={{ textAlign: 'center', marginBottom: '36px' }}>
              <h2 style={{ fontSize: '34px', fontWeight: 700, color: 'var(--fg)', margin: '0 0 12px', lineHeight: 1.2 }}>
                Transform Videos into{' '}
                <span className="gradient-text">Static Summaries</span>
              </h2>
              <p style={{ fontSize: '16px', color: 'var(--muted)', maxWidth: '580px', margin: '0 auto', lineHeight: 1.5 }}>
                Upload a video and our AI-powered pipeline will extract the most representative keyframes using advanced clustering algorithms.
              </p>
            </div>

            {/* Upload Component */}
            <VideoUpload
              onUpload={handleUpload}
              isUploading={isUploading}
              supportedFormats={config?.supported_formats || ['.mp4', '.avi', '.mov', '.mkv', '.webm']}
              maxSizeMB={config?.max_video_size_mb || 500}
            />

            {/* Features Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginTop: '48px' }}>
              <div className="feature-card">
                <div style={{
                  width: '48px', height: '48px', borderRadius: '12px', margin: '0 auto 14px',
                  background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(6, 182, 212, 0.15))',
                  border: '1px solid rgba(59, 130, 246, 0.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  <Film size={24} color="#3b82f6" />
                </div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--fg)', margin: '0 0 6px' }}>Smart Extraction</h3>
                <p style={{ fontSize: '13px', color: 'var(--muted)', margin: 0, lineHeight: 1.5 }}>
                  Redundant frames filtered using histogram analysis
                </p>
              </div>

              <div className="feature-card">
                <div style={{
                  width: '48px', height: '48px', borderRadius: '12px', margin: '0 auto 14px',
                  background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(236, 72, 153, 0.15))',
                  border: '1px solid rgba(168, 85, 247, 0.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  <Cpu size={24} color="#a855f7" />
                </div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--fg)', margin: '0 0 6px' }}>K-Means Clustering</h3>
                <p style={{ fontSize: '13px', color: 'var(--muted)', margin: 0, lineHeight: 1.5 }}>
                  Frames grouped by visual similarity
                </p>
              </div>

              <div className="feature-card">
                <div style={{
                  width: '48px', height: '48px', borderRadius: '12px', margin: '0 auto 14px',
                  background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.15))',
                  border: '1px solid rgba(34, 197, 94, 0.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  <LayoutGrid size={24} color="#22c55e" />
                </div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--fg)', margin: '0 0 6px' }}>Storyboard Output</h3>
                <p style={{ fontSize: '13px', color: 'var(--muted)', margin: 0, lineHeight: 1.5 }}>
                  Individual keyframes and grid layout
                </p>
              </div>
            </div>
          </>
        )}

        {state === 'processing' && jobStatus && <ProcessingProgress job={jobStatus} />}
        {state === 'completed' && summary && <SummaryGallery summary={summary} />}

        {state === 'error' && (
          <div style={{ maxWidth: '400px', margin: '0 auto', textAlign: 'center', padding: '32px 0' }}>
            <div style={{
              width: '56px', height: '56px', borderRadius: '14px', margin: '0 auto 16px',
              background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '24px'
            }}>❌</div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--fg)', margin: '0 0 8px' }}>Processing Failed</h2>
            <p style={{ color: 'var(--muted)', marginBottom: '16px', fontSize: '13px' }}>{error}</p>
            <button onClick={handleReset} className="btn-primary">Try Again</button>
          </div>
        )}
      </main>

      {/* History Modal */}
      <HistoryModal
        isOpen={showHistory}
        onClose={() => setShowHistory(false)}
        onSelectJob={handleSelectFromHistory}
      />
    </div>
  );
}
