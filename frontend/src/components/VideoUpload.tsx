'use client';

import { useState, useRef, useCallback } from 'react';
import { Upload, Film, AlertCircle, CheckCircle } from 'lucide-react';
import { FileVideo } from '@phosphor-icons/react';

interface VideoUploadProps {
    onUpload: (file: File) => void;
    isUploading: boolean;
    supportedFormats: string[];
    maxSizeMB: number;
}

export default function VideoUpload({ onUpload, isUploading, supportedFormats, maxSizeMB }: VideoUploadProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const validateFile = useCallback((file: File): string | null => {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!supportedFormats.includes(ext)) return `Unsupported format. Allowed: ${supportedFormats.join(', ')}`;
        if (file.size > maxSizeMB * 1024 * 1024) return `File too large. Maximum: ${maxSizeMB}MB`;
        return null;
    }, [supportedFormats, maxSizeMB]);

    const handleFile = useCallback((file: File) => {
        const err = validateFile(file);
        if (err) { setError(err); setSelectedFile(null); return; }
        setError(null);
        setSelectedFile(file);
    }, [validateFile]);

    const formatSize = (bytes: number) => bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

    return (
        <div style={{ maxWidth: '550px', margin: '0 auto' }}>
            {/* Drop Zone */}
            <div
                className="drop-zone"
                style={{
                    padding: '36px 24px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    borderColor: selectedFile ? 'var(--success)' : isDragging ? 'var(--primary)' : undefined,
                    background: selectedFile ? 'rgba(34, 197, 94, 0.05)' : isDragging ? 'rgba(99, 102, 241, 0.05)' : undefined
                }}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                onDrop={(e) => { e.preventDefault(); setIsDragging(false); e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]); }}
                onClick={() => fileInputRef.current?.click()}
            >
                <input ref={fileInputRef} type="file" accept={supportedFormats.join(',')} onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} style={{ display: 'none' }} />

                {selectedFile ? (
                    <>
                        <div style={{
                            width: '60px', height: '60px', borderRadius: '14px', margin: '0 auto 14px',
                            background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}>
                            <FileVideo size={30} weight="duotone" color="#22c55e" />
                        </div>
                        <p style={{ fontSize: '16px', fontWeight: 600, color: 'var(--fg)', margin: '0 0 4px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                            {selectedFile.name} <CheckCircle size={16} color="#22c55e" />
                        </p>
                        <p style={{ color: 'var(--muted)', fontSize: '13px', margin: 0 }}>{formatSize(selectedFile.size)}</p>
                    </>
                ) : (
                    <>
                        <div style={{
                            width: '60px', height: '60px', borderRadius: '14px', margin: '0 auto 14px',
                            background: isDragging ? 'var(--primary)' : 'var(--card)',
                            border: isDragging ? 'none' : '1px solid var(--border)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            transition: 'all 0.3s ease',
                            boxShadow: isDragging ? '0 0 32px rgba(99, 102, 241, 0.5)' : 'none'
                        }}>
                            <Upload size={28} color={isDragging ? 'white' : 'var(--muted)'} />
                        </div>
                        <p style={{ fontSize: '16px', fontWeight: 500, color: 'var(--fg)', margin: '0 0 4px' }}>Drop your video here</p>
                        <p style={{ color: 'var(--muted)', fontSize: '13px', margin: 0 }}>or click to browse</p>
                    </>
                )}

                <div style={{ marginTop: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', color: 'var(--muted)', fontSize: '11px' }}>
                    <Film size={12} />
                    <span>Supports: {supportedFormats.join(', ')} • Max: {maxSizeMB}MB</span>
                </div>
            </div>

            {/* Error */}
            {error && (
                <div style={{ marginTop: '12px', padding: '10px 14px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AlertCircle size={16} color="#ef4444" />
                    <p style={{ color: '#ef4444', fontSize: '12px', margin: 0 }}>{error}</p>
                </div>
            )}

            {/* Upload Button */}
            {selectedFile && !error && (
                <button onClick={() => onUpload(selectedFile)} disabled={isUploading} className="btn-primary" style={{ width: '100%', marginTop: '14px' }}>
                    {isUploading ? (
                        <>
                            <div style={{ width: '16px', height: '16px', border: '2px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                            Uploading...
                        </>
                    ) : (
                        <><Upload size={16} /> Start Processing</>
                    )}
                </button>
            )}

            <style jsx>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
    );
}
