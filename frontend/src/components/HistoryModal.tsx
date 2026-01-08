'use client';

import { useState, useEffect } from 'react';
import { X, Clock, CheckCircle, XCircle, Loader2, Play, Trash2, Layers, Hash, Download, Zap, Sparkles, TrendingUp, Info } from 'lucide-react';
import { api } from '@/utils/api';
import { JobListItem, STAGE_INFO } from '@/types';
import InteractiveTimeline from './InteractiveTimeline';

interface HistoryModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSelectJob: (jobId: number) => void;
}

export default function HistoryModal({ isOpen, onClose, onSelectJob }: HistoryModalProps) {
    const [jobs, setJobs] = useState<JobListItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!isOpen) return;

        setLoading(true);
        setError(null);

        api.listJobs()
            .then((data) => {
                setJobs(data.jobs || []);
                setLoading(false);
            })
            .catch((err) => {
                setError(err.message || 'Failed to load history');
                setLoading(false);
            });
    }, [isOpen]);

    if (!isOpen) return null;

    const getStatusIcon = (status: string) => {
        if (status === 'completed') return <CheckCircle size={18} color="#22c55e" />;
        if (status === 'failed') return <XCircle size={18} color="#ef4444" />;
        return <Loader2 size={18} color="#6366f1" className="animate-spin" />;
    };

    const getStatusColor = (status: string) => {
        if (status === 'completed') return { bg: 'rgba(34,197,94,0.1)', border: 'rgba(34,197,94,0.3)', text: '#22c55e' };
        if (status === 'failed') return { bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.3)', text: '#ef4444' };
        return { bg: 'rgba(99,102,241,0.1)', border: 'rgba(99,102,241,0.3)', text: '#6366f1' };
    };

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const formatDuration = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div
            style={{
                position: 'fixed', inset: 0, zIndex: 100,
                background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(8px)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px'
            }}
            onClick={onClose}
        >
            <div
                style={{
                    width: '100%', maxWidth: '600px', maxHeight: '75vh',
                    background: '#16161a', border: '1px solid #2a2a32', borderRadius: '14px',
                    display: 'flex', flexDirection: 'column', overflow: 'hidden'
                }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div style={{
                    padding: '14px 18px', borderBottom: '1px solid #2a2a32',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Clock size={18} color="#6366f1" />
                        <h2 style={{ fontSize: '16px', fontWeight: 600, color: '#f5f5f7', margin: 0 }}>
                            Processing History
                        </h2>
                    </div>
                    <button
                        onClick={onClose}
                        style={{
                            width: '32px', height: '32px', borderRadius: '8px',
                            background: '#1e1e24', border: '1px solid #2a2a32',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            cursor: 'pointer', transition: 'all 0.2s'
                        }}
                    >
                        <X size={16} color="#71717a" />
                    </button>
                </div>

                {/* Content */}
                <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
                    {loading && (
                        <div style={{ textAlign: 'center', padding: '32px 0' }}>
                            <Loader2 size={28} color="#6366f1" style={{ animation: 'spin 1s linear infinite' }} />
                            <p style={{ color: '#71717a', marginTop: '10px', fontSize: '13px' }}>Loading history...</p>
                        </div>
                    )}

                    {error && (
                        <div style={{
                            textAlign: 'center', padding: '32px 0',
                            color: '#ef4444'
                        }}>
                            <XCircle size={28} />
                            <p style={{ marginTop: '10px', fontSize: '13px' }}>{error}</p>
                        </div>
                    )}

                    {!loading && !error && jobs.length === 0 && (
                        <div style={{ textAlign: 'center', padding: '32px 0' }}>
                            <Clock size={32} color="#2a2a32" />
                            <p style={{ color: '#71717a', marginTop: '10px', fontSize: '13px' }}>
                                No processing history yet
                            </p>
                            <p style={{ color: '#52525b', fontSize: '11px', marginTop: '4px' }}>
                                Upload a video to get started
                            </p>
                        </div>
                    )}

                    {!loading && !error && jobs.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {jobs.map((job) => {
                                const statusColors = getStatusColor(job.status);
                                const isClickable = job.status === 'completed';

                                return (
                                    <div
                                        key={job.id}
                                        onClick={() => isClickable && onSelectJob(job.id)}
                                        style={{
                                            background: '#1e1e24', border: '1px solid #2a2a32', borderRadius: '10px',
                                            padding: '12px 14px', cursor: isClickable ? 'pointer' : 'default',
                                            transition: 'all 0.2s',
                                            ...(isClickable ? {} : { opacity: 0.7 })
                                        }}
                                        onMouseEnter={(e) => {
                                            if (isClickable) {
                                                e.currentTarget.style.borderColor = '#6366f1';
                                                e.currentTarget.style.background = '#252530';
                                            }
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.borderColor = '#2a2a32';
                                            e.currentTarget.style.background = '#1e1e24';
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                            {/* Status Icon */}
                                            <div style={{
                                                width: '36px', height: '36px', borderRadius: '8px',
                                                background: statusColors.bg, border: `1px solid ${statusColors.border}`,
                                                display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                                            }}>
                                                {getStatusIcon(job.status)}
                                            </div>

                                            {/* Info */}
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <p style={{
                                                    fontSize: '13px', fontWeight: 500, color: '#f5f5f7', margin: 0,
                                                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'
                                                }}>
                                                    {job.filename}
                                                </p>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                                                    <span style={{ fontSize: '11px', color: '#71717a' }}>
                                                        {formatDate(job.created_at)}
                                                    </span>
                                                    <span style={{ fontSize: '11px', color: '#52525b' }}>•</span>
                                                    <span style={{ fontSize: '11px', color: '#71717a' }}>
                                                        {formatDuration(job.duration)}
                                                    </span>
                                                </div>
                                            </div>

                                            {/* Status Badge */}
                                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px' }}>
                                                <div style={{
                                                    padding: '4px 10px', borderRadius: '6px', fontSize: '10px', fontWeight: 500,
                                                    background: statusColors.bg, color: statusColors.text, flexShrink: 0
                                                }}>
                                                    {STAGE_INFO[job.status]?.label || job.status}
                                                </div>

                                                {job.status === 'completed' && (
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                        {job.summary_confidence && (
                                                            <span style={{
                                                                fontSize: '9px',
                                                                color: job.summary_confidence === 'HIGH' ? '#22c55e' : job.summary_confidence === 'LOW' ? '#ef4444' : '#eab308',
                                                                background: 'rgba(255,255,255,0.05)',
                                                                padding: '2px 6px',
                                                                borderRadius: '4px',
                                                                border: '1px solid rgba(255,255,255,0.1)'
                                                            }}>
                                                                {job.summary_confidence}
                                                            </span>
                                                        )}
                                                        {job.compression_ratio && (
                                                            <span style={{ fontSize: '9px', color: '#71717a' }}>
                                                                {job.compression_ratio.toFixed(1)}x
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                            </div>

                                            {/* Action */}
                                            {isClickable && (
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                    <Play size={14} color="#6366f1" />
                                                </div>
                                            )}
                                        </div>

                                        {/* Rich Stats Grid - Mirroring SummaryGallery.tsx */}
                                        {job.status === 'completed' && (
                                            <div style={{ marginTop: '14px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px' }}>
                                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginBottom: '14px' }}>
                                                    {[
                                                        { icon: <Layers size={11} color="#6366f1" />, label: 'Keys', value: job.num_keyframes || 0 },
                                                        { icon: <Hash size={11} color="#a855f7" />, label: 'Orig', value: job.total_frames_original?.toLocaleString() || 'N/A' },
                                                        { icon: <Download size={11} color="#22c55e" />, label: 'Comp', value: job.total_frames_original ? `${((1 - (job.num_keyframes || 0) / job.total_frames_original) * 100).toFixed(1)}%` : 'N/A' },
                                                        { icon: <Zap size={11} color="#f97316" />, label: 'Pacing', value: job.video_pacing?.toUpperCase() || 'NORMAL' },
                                                    ].map((stat, i) => (
                                                        <div key={i} style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '6px', padding: '6px 8px' }}>
                                                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '2px' }}>
                                                                {stat.icon}
                                                                <span style={{ fontSize: '9px', color: '#71717a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{stat.label}</span>
                                                            </div>
                                                            <p style={{ fontSize: '11px', fontWeight: 600, color: '#f5f5f7', margin: 0 }}>{stat.value}</p>
                                                        </div>
                                                    ))}
                                                </div>

                                                {/* Visual Interactive Timeline Track */}
                                                <div style={{
                                                    margin: '20px 10px 30px',
                                                    padding: '10px 0',
                                                    pointerEvents: 'none', // Keep list clickable but let it show the visual
                                                    opacity: 0.9
                                                }}>
                                                    <InteractiveTimeline
                                                        summary={{
                                                            job_id: job.id,
                                                            video_duration: job.duration,
                                                            keyframes: job.keyframes || []
                                                        }}
                                                        compact={true}
                                                    />
                                                </div>

                                                {/* Summary Reason / Advanced Metrics Snippet */}
                                                {(job.compression_ratio || job.summary_reason) && (
                                                    <div style={{
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        gap: '8px',
                                                        fontSize: '10px',
                                                        color: '#a1a1aa',
                                                        background: 'rgba(99,102,241,0.03)',
                                                        padding: '6px 10px',
                                                        borderRadius: '6px',
                                                        border: '1px solid rgba(99,102,241,0.1)'
                                                    }}>
                                                        <Info size={12} color="#6366f1" />
                                                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                            {job.summary_reason || `Optimized with ${job.compression_ratio?.toFixed(1)}x compression ratio.`}
                                                        </span>
                                                        {job.temporal_coverage_score && (
                                                            <span style={{ color: '#22c55e', fontWeight: 600 }}>
                                                                {Math.round(job.temporal_coverage_score * 100)}% Coverage
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {/* Progress for in-progress jobs */}
                                        {job.status !== 'completed' && job.status !== 'failed' && (
                                            <div style={{ marginTop: '12px' }}>
                                                <div style={{ height: '4px', background: '#2a2a32', borderRadius: '2px', overflow: 'hidden' }}>
                                                    <div style={{
                                                        height: '100%', width: `${job.progress}%`,
                                                        background: 'linear-gradient(90deg, #6366f1, #a855f7)',
                                                        borderRadius: '2px', transition: 'width 0.3s'
                                                    }} />
                                                </div>
                                                <p style={{ fontSize: '11px', color: '#71717a', marginTop: '6px' }}>
                                                    {job.current_stage} • {Math.round(job.progress)}%
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>

            <style jsx>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
        </div>
    );
}
