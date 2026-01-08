'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import Image from 'next/image';
import { Play, Pause, Volume2, VolumeX, Maximize, Clock } from 'lucide-react';
import { SummaryResponse, KeyframeInfo } from '@/types';
import { api } from '@/utils/api';

interface InteractiveTimelineProps {
    summary: SummaryResponse | any;
    onSelectKeyframe?: (kf: KeyframeInfo) => void;
    compact?: boolean;
}

export default function InteractiveTimeline({ summary, onSelectKeyframe, compact = false }: InteractiveTimelineProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const timelineRef = useRef<HTMLDivElement>(null);

    const [isPlaying, setIsPlaying] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [selectedKeyframe, setSelectedKeyframe] = useState<KeyframeInfo | null>(null);
    const [hoveredKeyframe, setHoveredKeyframe] = useState<KeyframeInfo | null>(null);

    // Format time in MM:SS
    const formatTime = useCallback((seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }, []);

    // Update current time on video timeupdate
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleTimeUpdate = () => {
            setCurrentTime(video.currentTime);
        };

        const handlePlay = () => setIsPlaying(true);
        const handlePause = () => setIsPlaying(false);
        const handleEnded = () => setIsPlaying(false);

        video.addEventListener('timeupdate', handleTimeUpdate);
        video.addEventListener('play', handlePlay);
        video.addEventListener('pause', handlePause);
        video.addEventListener('ended', handleEnded);

        return () => {
            video.removeEventListener('timeupdate', handleTimeUpdate);
            video.removeEventListener('play', handlePlay);
            video.removeEventListener('pause', handlePause);
            video.removeEventListener('ended', handleEnded);
        };
    }, []);

    // Seek to keyframe timestamp
    const handleKeyframeClick = useCallback((keyframe: KeyframeInfo) => {
        if (videoRef.current) {
            videoRef.current.currentTime = keyframe.timestamp;
            setSelectedKeyframe(keyframe);
            if (onSelectKeyframe) onSelectKeyframe(keyframe);
        }
    }, [onSelectKeyframe]);

    // Click on timeline to seek
    const handleTimelineClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        if (!timelineRef.current || !videoRef.current) return;

        const rect = timelineRef.current.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percentage = clickX / rect.width;
        const seekTime = percentage * summary.video_duration;

        videoRef.current.currentTime = seekTime;
    }, [summary.video_duration]);

    // Toggle play/pause
    const togglePlay = useCallback(() => {
        if (videoRef.current) {
            if (isPlaying) {
                videoRef.current.pause();
            } else {
                videoRef.current.play();
            }
        }
    }, [isPlaying]);

    // Toggle mute
    const toggleMute = useCallback(() => {
        if (videoRef.current) {
            videoRef.current.muted = !isMuted;
            setIsMuted(!isMuted);
        }
    }, [isMuted]);

    // Fullscreen
    const handleFullscreen = useCallback(() => {
        if (videoRef.current) {
            if (videoRef.current.requestFullscreen) {
                videoRef.current.requestFullscreen();
            }
        }
    }, []);

    // Calculate playhead position
    const playheadPosition = (currentTime / summary.video_duration) * 100;

    return (
        <div className={compact ? "timeline-container-compact" : "glass timeline-container"} style={{
            padding: compact ? '0' : '20px',
            marginBottom: compact ? '10px' : '24px',
            background: compact ? 'transparent' : undefined,
            border: compact ? 'none' : undefined,
            boxShadow: compact ? 'none' : undefined
        }}>
            {/* Header - Hidden in compact mode */}
            {!compact && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                    <div style={{
                        width: '40px', height: '40px', borderRadius: '10px',
                        background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15))',
                        border: '1px solid rgba(99, 102, 241, 0.3)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                        <Clock size={20} color="#6366f1" />
                    </div>
                    <div>
                        <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--fg)', margin: 0 }}>
                            Interactive Timeline
                        </h3>
                        <p style={{ fontSize: '12px', color: 'var(--muted)', margin: '2px 0 0' }}>
                            Click a keyframe to jump to that moment
                        </p>
                    </div>
                </div>
            )}

            {/* Video Player - Hidden in compact mode */}
            {!compact && (
                <div style={{
                    position: 'relative',
                    borderRadius: '12px',
                    overflow: 'hidden',
                    background: '#000',
                    marginBottom: '16px'
                }}>
                    <video
                        ref={videoRef}
                        src={api.getVideoUrl(summary.job_id)}
                        style={{ width: '100%', display: 'block', maxHeight: '400px' }}
                        playsInline
                    />

                    {/* Video Controls Overlay */}
                    <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        padding: '12px 16px',
                        background: 'linear-gradient(transparent, rgba(0,0,0,0.8))',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px'
                    }}>
                        <button
                            onClick={togglePlay}
                            className="timeline-control-btn"
                            aria-label={isPlaying ? 'Pause' : 'Play'}
                        >
                            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        </button>

                        <span style={{ color: 'white', fontSize: '13px', fontFamily: 'monospace' }}>
                            {formatTime(currentTime)} / {formatTime(summary.video_duration)}
                        </span>

                        <div style={{ flex: 1 }} />

                        <button
                            onClick={toggleMute}
                            className="timeline-control-btn"
                            aria-label={isMuted ? 'Unmute' : 'Mute'}
                        >
                            {isMuted ? <VolumeX size={18} /> : <Volume2 size={18} />}
                        </button>

                        <button
                            onClick={handleFullscreen}
                            className="timeline-control-btn"
                            aria-label="Fullscreen"
                        >
                            <Maximize size={18} />
                        </button>
                    </div>
                </div>
            )}

            {/* Timeline Track */}
            <div
                ref={timelineRef}
                className={compact ? "timeline-track timeline-track-compact" : "timeline-track"}
                onClick={handleTimelineClick}
                style={compact ? { height: '2px', margin: '40px 0' } : {}}
            >
                {/* Progress fill */}
                <div
                    className="timeline-progress"
                    style={{ width: `${playheadPosition}%` }}
                />

                {/* Playhead */}
                {!compact && (
                    <div
                        className="timeline-playhead"
                        style={{ left: `${playheadPosition}%` }}
                    />
                )}


                {/* Keyframe Markers - Enhanced with importance */}
                {summary.keyframes?.map((kf: KeyframeInfo) => {
                    const position = (kf.timestamp / summary.video_duration) * 100;
                    const isSelected = selectedKeyframe?.index === kf.index;
                    const isHovered = hoveredKeyframe?.index === kf.index;
                    const confidence = kf.importance_confidence;

                    // Determine styling based on confidence
                    const isHigh = confidence === 'HIGH';
                    const isLow = confidence === 'LOW';

                    // Stagger logic: alternate markers up and down
                    const isEven = kf.index % 2 === 0;
                    const verticalOffset = isEven ? '-38px' : '38px';

                    return (
                        <div
                            key={kf.index}
                            className={`timeline-marker ${isSelected ? 'timeline-marker-active' : ''}`}
                            style={{
                                left: `${position}%`,
                                top: '50%',
                                opacity: isLow ? 0.5 : 1,
                                transform: `translate(-50%, calc(-50% + ${verticalOffset})) ${isSelected || isHovered ? 'scale(1.15)' : isHigh ? 'scale(1.1)' : 'scale(1)'}`,
                                zIndex: isSelected || isHovered ? 20 : isHigh ? 10 : isLow ? 1 : 5
                            }}
                            onClick={(e) => {
                                e.stopPropagation();
                                handleKeyframeClick(kf);
                            }}
                            onMouseEnter={() => setHoveredKeyframe(kf)}
                            onMouseLeave={() => setHoveredKeyframe(null)}
                        >
                            {/* Vertical connecting line */}
                            <div style={{
                                position: 'absolute',
                                left: '50%',
                                bottom: isEven ? '-100%' : 'auto',
                                top: isEven ? 'auto' : '-100%',
                                width: '1px',
                                height: '30px',
                                background: isHigh ? 'var(--success)' : 'rgba(255,255,255,0.2)',
                                transform: 'translateX(-50%)',
                                pointerEvents: 'none'
                            }} />

                            <div
                                className="timeline-marker-thumb"
                                style={{
                                    boxShadow: isHigh
                                        ? '0 0 12px rgba(34,197,94,0.6)'
                                        : isLow
                                            ? 'none'
                                            : '0 2px 8px rgba(0,0,0,0.3)',
                                    border: isHigh
                                        ? '2px solid #22c55e'
                                        : isLow
                                            ? '1px solid rgba(255,255,255,0.1)'
                                            : '2px solid rgba(99,102,241,0.5)'
                                }}
                            >
                                <Image
                                    src={api.getKeyframeUrl(summary.job_id, kf.index)}
                                    alt={`Keyframe ${kf.index}`}
                                    fill
                                    style={{ objectFit: 'cover' }}
                                    sizes="60px"
                                    unoptimized
                                />
                            </div>

                            {/* Enhanced Tooltip with importance */}
                            {(isHovered || isSelected) && (
                                <div className="timeline-marker-tooltip" style={{ minWidth: '100px' }}>
                                    <span style={{ fontWeight: 600 }}>#{kf.index}</span>
                                    <span>{kf.timestamp_formatted}</span>
                                    {kf.importance_score !== undefined && (
                                        <span style={{
                                            color: isHigh ? '#22c55e' : isLow ? '#ef4444' : '#eab308',
                                            fontWeight: 600
                                        }}>
                                            {(kf.importance_score * 100).toFixed(0)}% {confidence}
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Time Labels */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--muted)',
                fontFamily: 'monospace'
            }}>
                <span>00:00</span>
                <span>{formatTime(summary.video_duration)}</span>
            </div>
        </div>
    );
}
