'use client';

import { JobStatusResponse, STAGE_INFO, JobStatus } from '@/types';
import { Clock, Upload, Film, Filter, Cpu, Sliders, Target, Box, Image, Layout, CheckCircle, XCircle } from 'lucide-react';

interface ProcessingProgressProps {
    job: JobStatusResponse;
}

const STAGE_ICONS: Record<string, React.ReactNode> = {
    pending: <Clock size={18} />,
    uploaded: <Upload size={18} />,
    extracting_frames: <Film size={18} />,
    filtering_redundancy: <Filter size={18} />,
    extracting_features: <Cpu size={18} />,
    normalizing: <Sliders size={18} />,
    finding_optimal_k: <Target size={18} />,
    clustering: <Box size={18} />,
    selecting_keyframes: <Image size={18} />,
    generating_summary: <Layout size={18} />,
    completed: <CheckCircle size={18} />,
    failed: <XCircle size={18} />,
};

const STAGES: JobStatus[] = [
    'extracting_frames', 'filtering_redundancy', 'extracting_features', 'normalizing',
    'finding_optimal_k', 'clustering', 'selecting_keyframes', 'generating_summary',
];

export default function ProcessingProgress({ job }: ProcessingProgressProps) {
    const currentStageIndex = STAGES.indexOf(job.status as JobStatus);
    const isCompleted = job.status === 'completed';
    const isFailed = job.status === 'failed';

    return (
        <div style={{ maxWidth: '650px', margin: '0 auto' }}>
            {/* Header Card */}
            <div className="glass" style={{ padding: '20px', marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px' }}>
                    <div>
                        <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--fg)', margin: 0 }}>
                            {job.video_filename || 'Processing Video'}
                        </h2>
                        <p style={{ color: 'var(--muted)', fontSize: '13px', marginTop: '4px' }}>
                            {job.video_duration ? `Duration: ${Math.floor(job.video_duration / 60)}m ${Math.floor(job.video_duration % 60)}s` : 'Analyzing...'}
                        </p>
                    </div>
                    <div style={{
                        padding: '8px 14px', borderRadius: '8px', fontSize: '12px', fontWeight: 500,
                        background: isCompleted ? 'rgba(34,197,94,0.1)' : isFailed ? 'rgba(239,68,68,0.1)' : 'rgba(99,102,241,0.1)',
                        color: isCompleted ? '#22c55e' : isFailed ? '#ef4444' : '#6366f1',
                        border: `1px solid ${isCompleted ? 'rgba(34,197,94,0.3)' : isFailed ? 'rgba(239,68,68,0.3)' : 'rgba(99,102,241,0.3)'}`
                    }}>
                        {STAGE_INFO[job.status]?.label || job.status}
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="progress-bar">
                    <div className="progress-bar-fill" style={{ width: `${job.progress}%` }} />
                </div>
                <p style={{ textAlign: 'right', color: 'var(--muted)', fontSize: '12px', fontWeight: 500, marginTop: '6px' }}>{Math.round(job.progress)}%</p>

                {job.stage_detail && (
                    <p style={{ marginTop: '10px', padding: '10px', borderRadius: '6px', background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--fg)', fontSize: '12px' }}>
                        {job.stage_detail}
                    </p>
                )}

                {isFailed && job.error_message && (
                    <div style={{ marginTop: '10px', padding: '10px', borderRadius: '6px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#ef4444', fontSize: '12px' }}>
                        {job.error_message}
                    </div>
                )}
            </div>

            {/* Pipeline Stages */}
            <div className="glass" style={{ padding: '20px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--fg)', margin: '0 0 16px' }}>Pipeline Stages</h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {STAGES.map((stage, index) => {
                        const stageInfo = STAGE_INFO[stage];
                        const isActive = stage === job.status;
                        const isComplete = currentStageIndex > index || isCompleted;

                        return (
                            <div key={stage} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <div style={{
                                    width: '36px', height: '36px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'all 0.3s',
                                    background: isComplete ? '#22c55e' : isActive ? '#6366f1' : 'var(--card)',
                                    color: (isComplete || isActive) ? 'white' : 'var(--muted)',
                                    border: (isComplete || isActive) ? 'none' : '1px solid var(--border)',
                                    boxShadow: isActive ? '0 0 20px rgba(99,102,241,0.4)' : 'none'
                                }}>
                                    {STAGE_ICONS[stage]}
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <p style={{ fontWeight: 500, color: (isComplete || isActive) ? 'var(--fg)' : 'var(--muted)', margin: 0, fontSize: '13px' }}>{stageInfo.label}</p>
                                    {isActive && job.stage_detail && <p style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '2px' }}>{job.stage_detail}</p>}
                                </div>
                                <div style={{ flexShrink: 0 }}>
                                    {isComplete && <CheckCircle size={18} color="#22c55e" />}
                                    {isActive && <div style={{ width: '18px', height: '18px', border: '2px solid rgba(99,102,241,0.3)', borderTopColor: '#6366f1', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Stats */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginTop: '18px', paddingTop: '16px', borderTop: '1px solid var(--border)' }}>
                    {[
                        { label: 'Frames Extracted', value: job.frames_extracted },
                        { label: 'After Filtering', value: job.frames_filtered },
                        { label: 'Clusters Found', value: job.clusters_found || '-' }
                    ].map((stat, i) => (
                        <div key={i} style={{ textAlign: 'center', padding: '12px', borderRadius: '10px', background: 'var(--card)', border: '1px solid var(--border)' }}>
                            <p className="gradient-text" style={{ fontSize: '24px', fontWeight: 700, margin: 0 }}>{stat.value}</p>
                            <p style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '4px' }}>{stat.label}</p>
                        </div>
                    ))}
                </div>
            </div>

            <style jsx>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
    );
}
