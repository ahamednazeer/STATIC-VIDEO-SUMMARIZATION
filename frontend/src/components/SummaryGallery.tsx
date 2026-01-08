'use client';

import { useState } from 'react';
import Image from 'next/image';
import { X, Clock, Hash, LayoutGrid, ChevronLeft, ChevronRight, Download, Layers, Sparkles, TrendingUp, Zap, Eye, Info } from 'lucide-react';
import { SummaryResponse, KeyframeInfo } from '@/types';
import { api } from '@/utils/api';
import InteractiveTimeline from './InteractiveTimeline';

interface SummaryGalleryProps {
    summary: SummaryResponse;
}

// Helper to get confidence color
const getConfidenceColor = (confidence?: string) => {
    switch (confidence) {
        case 'HIGH': return { bg: 'rgba(34,197,94,0.15)', border: 'rgba(34,197,94,0.4)', text: '#22c55e' };
        case 'MEDIUM': return { bg: 'rgba(234,179,8,0.15)', border: 'rgba(234,179,8,0.4)', text: '#eab308' };
        case 'LOW': return { bg: 'rgba(239,68,68,0.15)', border: 'rgba(239,68,68,0.4)', text: '#ef4444' };
        default: return { bg: 'rgba(99,102,241,0.15)', border: 'rgba(99,102,241,0.4)', text: '#6366f1' };
    }
};

// Helper to get top factors explanation
const getTopFactors = (factors?: KeyframeInfo['importance_factors']) => {
    if (!factors) return [];
    const entries = Object.entries(factors)
        .map(([key, value]) => ({ key, value: value || 0 }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 3);

    const factorLabels: Record<string, string> = {
        representativeness: 'High cluster representativeness',
        dominance: 'Strong scene dominance',
        visual_richness: 'High visual richness',
        temporal_coverage: 'Good temporal coverage',
        visual_novelty: 'Visual novelty',
        motion_context: 'Motion context'
    };

    return entries.map(e => ({
        label: factorLabels[e.key] || e.key,
        value: e.value
    }));
};

export default function SummaryGallery({ summary }: SummaryGalleryProps) {
    const [selectedKeyframe, setSelectedKeyframe] = useState<KeyframeInfo | null>(null);
    const [showStoryboard, setShowStoryboard] = useState(false);
    const [hoveredKeyframe, setHoveredKeyframe] = useState<number | null>(null);

    const navigateKeyframe = (dir: 'prev' | 'next') => {
        if (!selectedKeyframe) return;
        const idx = summary.keyframes.findIndex(k => k.index === selectedKeyframe.index);
        const newIdx = dir === 'prev' ? Math.max(0, idx - 1) : Math.min(summary.keyframes.length - 1, idx + 1);
        setSelectedKeyframe(summary.keyframes[newIdx]);
    };

    return (
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
            {/* Header */}
            <div className="glass" style={{ padding: '20px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '16px' }}>
                    <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '24px' }}>✨</div>
                    <div style={{ flex: 1 }}>
                        <h2 className="gradient-text" style={{ fontSize: '22px', fontWeight: 700, margin: 0 }}>Summary Generated!</h2>
                        <p style={{ color: 'var(--muted)', fontSize: '13px', marginTop: '2px' }}>{summary.video_filename} • {summary.num_keyframes} keyframes</p>
                    </div>
                    {/* Summary Confidence Badge */}
                    {summary.summary_confidence && (
                        <div style={{
                            ...getConfidenceColor(summary.summary_confidence),
                            background: getConfidenceColor(summary.summary_confidence).bg,
                            border: `1px solid ${getConfidenceColor(summary.summary_confidence).border}`,
                            color: getConfidenceColor(summary.summary_confidence).text,
                            padding: '6px 12px',
                            borderRadius: '8px',
                            fontSize: '12px',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px'
                        }}>
                            <Sparkles size={14} />
                            {summary.summary_confidence} Confidence
                        </div>
                    )}
                </div>

                {/* Stats - Enhanced with pacing */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px' }}>
                    {[
                        { icon: <Layers size={16} color="#6366f1" />, label: 'Keyframes', value: summary.num_keyframes },
                        { icon: <Hash size={16} color="#a855f7" />, label: 'Original Frames', value: summary.total_frames_original.toLocaleString() },
                        { icon: <Clock size={16} color="#06b6d4" />, label: 'After Filtering', value: summary.frames_after_filtering.toLocaleString() },
                        { icon: <Download size={16} color="#22c55e" />, label: 'Compression', value: `${((1 - summary.num_keyframes / summary.total_frames_original) * 100).toFixed(1)}%`, color: '#22c55e' },
                        { icon: <Zap size={16} color="#f97316" />, label: 'Video Pacing', value: summary.video_pacing?.toUpperCase() || 'NORMAL', color: '#f97316' },
                    ].map((stat, i) => (
                        <div key={i} style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '10px', padding: '14px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>{stat.icon}<span style={{ fontSize: '11px', color: 'var(--muted)' }}>{stat.label}</span></div>
                            <p style={{ fontSize: '20px', fontWeight: 700, color: stat.color || 'var(--fg)', margin: 0 }}>{stat.value}</p>
                        </div>
                    ))}
                </div>

                <div style={{ display: 'flex', gap: '10px', marginTop: '16px' }}>
                    {summary.grid_available && <button onClick={() => setShowStoryboard(true)} className="btn-primary"><LayoutGrid size={16} /> View Storyboard</button>}
                    <a href={api.getDownloadAllUrl(summary.job_id)} download className="btn-secondary" style={{ textDecoration: 'none' }}><Download size={16} /> Download All</a>
                </div>

                {/* Summary Reason */}
                {summary.summary_reason && (
                    <div style={{
                        marginTop: '16px',
                        padding: '12px 16px',
                        background: 'linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1))',
                        border: '1px solid rgba(99,102,241,0.3)',
                        borderRadius: '10px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px'
                    }}>
                        <Info size={18} color="#6366f1" />
                        <span style={{ color: 'var(--fg)', fontSize: '13px' }}>{summary.summary_reason}</span>
                    </div>
                )}

                {/* Advanced Metrics */}
                {(summary.compression_ratio || summary.redundancy_removed || summary.temporal_coverage_score) && (
                    <div style={{ marginTop: '16px', padding: '16px', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '12px' }}>
                        <h4 style={{ fontSize: '13px', fontWeight: 600, color: 'var(--muted)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <TrendingUp size={14} /> ADVANCED METRICS
                        </h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                            {/* Compression Ratio */}
                            <div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                    <span style={{ fontSize: '11px', color: 'var(--muted)' }}>Compression Ratio</span>
                                    <span style={{ fontSize: '13px', fontWeight: 700, color: '#22c55e' }}>{summary.compression_ratio?.toFixed(1)}x</span>
                                </div>
                                <div style={{ height: '6px', background: 'rgba(34,197,94,0.2)', borderRadius: '3px', overflow: 'hidden' }}>
                                    <div style={{ width: `${Math.min(100, (summary.compression_ratio || 0) / 100 * 100)}%`, height: '100%', background: '#22c55e', borderRadius: '3px' }} />
                                </div>
                            </div>
                            {/* Redundancy Removed */}
                            <div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                    <span style={{ fontSize: '11px', color: 'var(--muted)' }}>Redundancy Removed</span>
                                    <span style={{ fontSize: '13px', fontWeight: 700, color: '#a855f7' }}>{((summary.redundancy_removed || 0) * 100).toFixed(1)}%</span>
                                </div>
                                <div style={{ height: '6px', background: 'rgba(168,85,247,0.2)', borderRadius: '3px', overflow: 'hidden' }}>
                                    <div style={{ width: `${(summary.redundancy_removed || 0) * 100}%`, height: '100%', background: '#a855f7', borderRadius: '3px' }} />
                                </div>
                            </div>
                            {/* Temporal Coverage */}
                            <div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                    <span style={{ fontSize: '11px', color: 'var(--muted)' }}>Temporal Coverage</span>
                                    <span style={{ fontSize: '13px', fontWeight: 700, color: '#06b6d4' }}>{((summary.temporal_coverage_score || 0) * 100).toFixed(0)}%</span>
                                </div>
                                <div style={{ height: '6px', background: 'rgba(6,182,212,0.2)', borderRadius: '3px', overflow: 'hidden' }}>
                                    <div style={{ width: `${(summary.temporal_coverage_score || 0) * 100}%`, height: '100%', background: '#06b6d4', borderRadius: '3px' }} />
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Interactive Timeline */}
            <InteractiveTimeline summary={summary} />

            {/* Keyframes Grid - Enhanced with importance */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '14px' }}>
                {summary.keyframes.map((kf) => {
                    const confidence = kf.importance_confidence;
                    const colors = getConfidenceColor(confidence);
                    const isHovered = hoveredKeyframe === kf.index;
                    const topFactors = getTopFactors(kf.importance_factors);

                    return (
                        <div
                            key={kf.index}
                            className="keyframe-card"
                            style={{
                                cursor: 'pointer',
                                position: 'relative',
                                opacity: confidence === 'LOW' ? 0.7 : 1,
                                transform: confidence === 'HIGH' ? 'scale(1.02)' : 'scale(1)',
                                transition: 'all 0.2s ease'
                            }}
                            onClick={() => setSelectedKeyframe(kf)}
                            onMouseEnter={() => setHoveredKeyframe(kf.index)}
                            onMouseLeave={() => setHoveredKeyframe(null)}
                        >
                            <div style={{ position: 'relative', aspectRatio: '16/9', background: 'var(--card)' }}>
                                <Image src={api.getKeyframeUrl(summary.job_id, kf.index)} alt={`Keyframe ${kf.index}`} fill style={{ objectFit: 'cover' }} sizes="20vw" unoptimized />

                                {/* Frame number badge */}
                                <div style={{ position: 'absolute', top: '6px', left: '6px', background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)', padding: '3px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 500, color: 'white' }}>#{kf.index}</div>

                                {/* Importance score badge (top right) */}
                                {kf.importance_score !== undefined && (
                                    <div style={{
                                        position: 'absolute',
                                        top: '6px',
                                        right: '6px',
                                        background: colors.bg,
                                        border: `1px solid ${colors.border}`,
                                        backdropFilter: 'blur(4px)',
                                        padding: '3px 8px',
                                        borderRadius: '6px',
                                        fontSize: '10px',
                                        fontWeight: 600,
                                        color: colors.text,
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '4px'
                                    }}>
                                        <TrendingUp size={10} />
                                        {(kf.importance_score * 100).toFixed(0)}%
                                    </div>
                                )}

                                {/* Confidence indicator bar at bottom of image */}
                                {confidence && (
                                    <div style={{
                                        position: 'absolute',
                                        bottom: 0,
                                        left: 0,
                                        right: 0,
                                        height: '3px',
                                        background: colors.text,
                                        opacity: 0.8
                                    }} />
                                )}
                            </div>

                            {/* Info section */}
                            <div style={{ padding: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--muted)' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Clock size={11} />{kf.timestamp_formatted}</span>
                                </div>
                                {confidence && (
                                    <span style={{
                                        fontSize: '9px',
                                        fontWeight: 600,
                                        color: colors.text,
                                        background: colors.bg,
                                        padding: '2px 6px',
                                        borderRadius: '4px'
                                    }}>
                                        {confidence}
                                    </span>
                                )}
                            </div>

                            {/* Context Snippet */}
                            {kf.context && (
                                <div style={{ padding: '0 10px 10px', fontSize: '10px', color: 'var(--muted)', fontStyle: 'italic', opacity: 0.8 }}>
                                    {kf.context.split('.')[0]}...
                                </div>
                            )}

                            {/* Tooltip on hover - "Why this frame?" */}
                            {isHovered && topFactors.length > 0 && (
                                <div style={{
                                    position: 'absolute',
                                    bottom: '100%',
                                    left: '50%',
                                    transform: 'translateX(-50%)',
                                    marginBottom: '8px',
                                    background: 'rgba(0,0,0,0.9)',
                                    backdropFilter: 'blur(8px)',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    borderRadius: '10px',
                                    padding: '12px',
                                    width: '200px',
                                    zIndex: 50,
                                    boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
                                }}>
                                    <div style={{ fontSize: '10px', fontWeight: 600, color: '#a855f7', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                        <Info size={10} />
                                        WHY THIS FRAME?
                                    </div>
                                    {topFactors.map((f, i) => (
                                        <div key={i} style={{ fontSize: '11px', color: 'white', marginBottom: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <span style={{ opacity: 0.8 }}>• {f.label}</span>
                                            <span style={{ color: '#22c55e', fontWeight: 600 }}>{(f.value * 100).toFixed(0)}%</span>
                                        </div>
                                    ))}
                                    {/* Arrow */}
                                    <div style={{
                                        position: 'absolute',
                                        bottom: '-6px',
                                        left: '50%',
                                        transform: 'translateX(-50%)',
                                        width: 0,
                                        height: 0,
                                        borderLeft: '6px solid transparent',
                                        borderRight: '6px solid transparent',
                                        borderTop: '6px solid rgba(0,0,0,0.9)'
                                    }} />
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Lightbox - Enhanced with importance info */}
            {selectedKeyframe && (
                <div
                    style={{
                        position: 'fixed',
                        inset: 0,
                        zIndex: 100,
                        background: 'rgba(0,0,0,0.95)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        padding: '60px 24px',
                        overflowY: 'auto',
                        cursor: 'zoom-out'
                    }}
                    onClick={() => setSelectedKeyframe(null)}
                >
                    <button onClick={(e) => { e.stopPropagation(); setSelectedKeyframe(null); }} style={{ position: 'fixed', top: '20px', right: '30px', width: '48px', height: '48px', borderRadius: '50%', background: 'rgba(255,255,255,0.2)', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 110, backdropFilter: 'blur(8px)' }}><X size={24} color="white" /></button>
                    <button onClick={(e) => { e.stopPropagation(); navigateKeyframe('prev'); }} style={{ position: 'fixed', left: '20px', top: '50%', transform: 'translateY(-50%)', width: '52px', height: '52px', borderRadius: '50%', background: 'rgba(255,255,255,0.1)', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 110 }}><ChevronLeft size={28} color="white" /></button>
                    <button onClick={(e) => { e.stopPropagation(); navigateKeyframe('next'); }} style={{ position: 'fixed', right: '20px', top: '50%', transform: 'translateY(-50%)', width: '52px', height: '52px', borderRadius: '50%', background: 'rgba(255,255,255,0.1)', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 110 }}><ChevronRight size={28} color="white" /></button>

                    <div style={{ maxWidth: '1000px', width: '100%', cursor: 'default' }} onClick={(e) => e.stopPropagation()}>
                        <div style={{ position: 'relative', aspectRatio: '16/9', borderRadius: '16px', overflow: 'hidden', background: 'black' }}>
                            <Image src={api.getKeyframeUrl(summary.job_id, selectedKeyframe.index)} alt={`Keyframe ${selectedKeyframe.index}`} fill style={{ objectFit: 'contain' }} quality={100} unoptimized />
                        </div>

                        {/* Context Banner */}
                        {selectedKeyframe.context && (
                            <div style={{
                                marginTop: '16px',
                                background: 'rgba(99,102,241,0.08)',
                                border: '1px solid rgba(99,102,241,0.2)',
                                borderRadius: '12px',
                                padding: '12px 20px',
                                textAlign: 'center'
                            }}>
                                <p style={{ margin: 0, color: 'rgba(255,255,255,0.9)', fontSize: '15px', fontWeight: 500, fontStyle: 'italic' }}>
                                    "{selectedKeyframe.context}"
                                </p>
                            </div>
                        )}

                        {/* Info badges */}
                        <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center', gap: '12px', flexWrap: 'wrap' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white', background: 'rgba(255,255,255,0.1)', padding: '10px 18px', borderRadius: '999px' }}><Hash size={16} />Keyframe {selectedKeyframe.index}</span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white', background: 'rgba(255,255,255,0.1)', padding: '10px 18px', borderRadius: '999px' }}><Clock size={16} />{selectedKeyframe.timestamp_formatted}</span>
                            {selectedKeyframe.importance_score !== undefined && (
                                <span style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    color: getConfidenceColor(selectedKeyframe.importance_confidence).text,
                                    background: getConfidenceColor(selectedKeyframe.importance_confidence).bg,
                                    border: `1px solid ${getConfidenceColor(selectedKeyframe.importance_confidence).border}`,
                                    padding: '10px 18px',
                                    borderRadius: '999px'
                                }}>
                                    <TrendingUp size={16} />
                                    Importance: {(selectedKeyframe.importance_score * 100).toFixed(0)}% ({selectedKeyframe.importance_confidence})
                                </span>
                            )}
                        </div>

                        {/* Deep Detail Analysis Dashboard */}
                        {selectedKeyframe.deep_analysis && (
                            <div style={{
                                marginTop: '30px',
                                background: 'rgba(255,255,255,0.03)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '20px',
                                padding: '24px',
                                width: '100%',
                                maxWidth: '800px',
                                margin: '30px auto 0 auto'
                            }}>
                                <h4 style={{ margin: '0 0 20px 0', color: 'var(--accent)', fontSize: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', flexWrap: 'wrap' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                        <Sparkles size={18} /> Deep Industry Analysis
                                    </div>
                                    {selectedKeyframe.domain && (
                                        <span style={{
                                            fontSize: '10px',
                                            background: 'rgba(255,255,255,0.1)',
                                            padding: '4px 12px',
                                            borderRadius: '999px',
                                            color: 'rgba(255,255,255,0.8)',
                                            border: '1px solid rgba(255,255,255,0.1)'
                                        }}>
                                            Domain: {selectedKeyframe.domain}
                                        </span>
                                    )}
                                </h4>

                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
                                    {/* Objects & Actions */}
                                    <div>
                                        <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'center' }}>Visual Content</p>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
                                            {selectedKeyframe.deep_analysis.objects.map((obj, i) => (
                                                <div key={`obj-${i}`} style={{ background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.3)', color: '#a5b4fc', padding: '4px 10px', borderRadius: '6px', fontSize: '12px' }}>
                                                    {obj.label}
                                                </div>
                                            ))}
                                            {selectedKeyframe.deep_analysis.actions.map((act, i) => (
                                                <div key={`act-${i}`} style={{ background: 'rgba(34,197,94,0.15)', border: '1px solid rgba(34,197,94,0.3)', color: '#86efac', padding: '4px 10px', borderRadius: '6px', fontSize: '12px' }}>
                                                    {act.label}
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Environment & Setting */}
                                    <div>
                                        <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'center' }}>Scene & Setting</p>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
                                            {selectedKeyframe.deep_analysis.environment.map((env, i) => {
                                                const isNature = selectedKeyframe.domain?.includes('Nature');
                                                const isCulinary = selectedKeyframe.domain?.includes('Culinary');
                                                const isIndustrial = selectedKeyframe.domain?.includes('Industrial');

                                                return (
                                                    <div
                                                        key={`env-${i}`}
                                                        style={{
                                                            background: isNature ? 'rgba(34,197,94,0.15)' : isCulinary ? 'rgba(249,115,22,0.15)' : isIndustrial ? 'rgba(234,179,8,0.15)' : 'rgba(249,115,22,0.15)',
                                                            border: `1px solid ${isNature ? 'rgba(34,197,94,0.3)' : isCulinary ? 'rgba(249,115,22,0.3)' : isIndustrial ? 'rgba(234,179,8,0.3)' : 'rgba(249,115,22,0.3)'}`,
                                                            color: isNature ? '#86efac' : isCulinary ? '#fdba74' : isIndustrial ? '#fde047' : '#fdba74',
                                                            padding: '4px 10px',
                                                            borderRadius: '6px',
                                                            fontSize: '12px'
                                                        }}
                                                    >
                                                        {env.label}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>

                                    {/* Visual Attributes */}
                                    <div>
                                        <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'center' }}>Visual Attributes</p>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
                                            {selectedKeyframe.deep_analysis.attributes.map((attr, i) => (
                                                <div key={`attr-${i}`} style={{ background: 'rgba(168,85,247,0.15)', border: '1px solid rgba(168,85,247,0.3)', color: '#d8b4fe', padding: '4px 10px', borderRadius: '6px', fontSize: '12px' }}>
                                                    {attr.label}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Factor breakdown */}
                        {selectedKeyframe.importance_factors && (
                            <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                {getTopFactors(selectedKeyframe.importance_factors).map((f, i) => (
                                    <span key={i} style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '6px',
                                        fontSize: '12px',
                                        color: 'rgba(255,255,255,0.7)',
                                        background: 'rgba(255,255,255,0.05)',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        padding: '6px 12px',
                                        borderRadius: '999px'
                                    }}>
                                        <Eye size={12} />
                                        {f.label}: <span style={{ color: '#22c55e' }}>{(f.value * 100).toFixed(0)}%</span>
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Storyboard */}
            {showStoryboard && (
                <div
                    style={{
                        position: 'fixed',
                        inset: 0,
                        zIndex: 100,
                        background: 'rgba(0,0,0,0.95)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        padding: '60px 24px',
                        overflowY: 'auto'
                    }}
                    onClick={() => setShowStoryboard(false)}
                >
                    <button onClick={() => setShowStoryboard(false)} style={{ position: 'fixed', top: '20px', right: '30px', width: '48px', height: '48px', borderRadius: '50%', background: 'rgba(255,255,255,0.2)', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 110, backdropFilter: 'blur(8px)' }}><X size={24} color="white" /></button>
                    <div style={{ maxWidth: '1200px', width: '100%', cursor: 'default' }} onClick={(e) => e.stopPropagation()}>
                        <div style={{ position: 'relative', aspectRatio: '16/9', borderRadius: '16px', overflow: 'hidden', background: 'black' }}>
                            <Image src={api.getStoryboardUrl(summary.job_id)} alt="Storyboard" fill style={{ objectFit: 'contain' }} quality={100} unoptimized />
                        </div>
                        <p style={{ textAlign: 'center', color: 'white', fontSize: '18px', marginTop: '20px' }}>Storyboard Grid • {summary.num_keyframes} Keyframes</p>
                    </div>
                </div>
            )}
        </div>
    );
}
