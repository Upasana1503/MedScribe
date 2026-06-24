import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Mic, Square, Play, Pause, Save, Loader2, AlertCircle, CheckCircle2, RotateCcw, FileText, Tag, ClipboardList, Upload, Stethoscope, Activity } from 'lucide-react';
import './index.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const ACCEPTED_AUDIO_TYPES = '.wav,.mp3,.m4a,.flac,.aac,.ogg,.wma,.webm';

const audioBufferToWavBlob = (audioBuffer) => {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const samples = audioBuffer.length;
  const bytesPerSample = 2;
  const blockAlign = channelCount * bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples * blockAlign);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples * blockAlign, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples * blockAlign, true);

  let offset = 44;
  const channels = Array.from({ length: channelCount }, (_, channel) => audioBuffer.getChannelData(channel));
  for (let i = 0; i < samples; i += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channels[channel][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += bytesPerSample;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
};

const convertRecordingToWav = async (recordingBlob) => {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error('Audio conversion is not supported in this browser.');
  }

  const arrayBuffer = await recordingBlob.arrayBuffer();
  const audioContext = new AudioContextClass();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBufferToWavBlob(audioBuffer);
  } finally {
    await audioContext.close();
  }
};

const ENTITY_LABELS = {
  medications: { label: 'Medications', color: '#3b82f6', icon: '💊' },
  conditions: { label: 'Conditions', color: '#ef4444', icon: '🔴' },
  symptoms: { label: 'Symptoms', color: '#f59e0b', icon: '⚡' },
  procedures: { label: 'Procedures', color: '#8b5cf6', icon: '🔬' },
  vitals: { label: 'Vitals', color: '#10b981', icon: '📊' },
  anatomy: { label: 'Anatomy', color: '#ec4899', icon: '🫀' },
  findings: { label: 'Findings', color: '#6366f1', icon: '🔍' },
  demographics: { label: 'Demographics', color: '#14b8a6', icon: '👤' },
  history: { label: 'History', color: '#f97316', icon: '📋' },
  temporal: { label: 'Temporal', color: '#64748b', icon: '🕐' },
  dosages: { label: 'Dosages', color: '#06b6d4', icon: '💉' },
};

const EntityBadge = ({ text, color }) => (
  <span style={{
    display: 'inline-block',
    padding: '0.3rem 0.75rem',
    borderRadius: '999px',
    fontSize: '0.8rem',
    fontWeight: '500',
    background: `${color}15`,
    color: color,
    border: `1px solid ${color}30`,
    margin: '0.2rem',
    transition: 'all 0.2s',
  }}>
    {text}
  </span>
);

const EntitiesPanel = ({ entities }) => {
  if (!entities || Object.keys(entities).length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '3rem 0' }}>
        <Tag size={48} color="var(--text-secondary)" style={{ opacity: 0.3, marginBottom: '1rem' }} />
        <p style={{ color: 'var(--text-secondary)' }}>No medical entities extracted yet.</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
      {Object.entries(entities).map(([category, items]) => {
        const config = ENTITY_LABELS[category] || { label: category, color: '#64748b', icon: '📌' };
        if (!items || items.length === 0) return null;
        return (
          <div key={category} style={{ background: 'var(--bg-secondary)', borderRadius: '12px', padding: '1rem 1.2rem' }}>
            <h4 style={{ fontSize: '0.8rem', fontWeight: '600', color: config.color, marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span>{config.icon}</span> {config.label}
              <span style={{ background: `${config.color}20`, color: config.color, padding: '0.1rem 0.5rem', borderRadius: '999px', fontSize: '0.7rem', marginLeft: '0.3rem' }}>{items.length}</span>
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.2rem' }}>
              {items.map((item, idx) => (
                <EntityBadge key={idx} text={item} color={config.color} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const SOAPSection = ({ title, letter, content, color }) => (
  <div style={{ marginBottom: '1rem' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
      <span style={{ width: '28px', height: '28px', borderRadius: '8px', background: `${color}20`, color: color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: '700', fontSize: '0.85rem' }}>
        {letter}
      </span>
      <h4 style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-primary)' }}>{title}</h4>
    </div>
    <div style={{ padding: '1rem 1.2rem', background: 'var(--bg-secondary)', borderRadius: '10px', borderLeft: `3px solid ${color}`, whiteSpace: 'pre-wrap', lineHeight: '1.7', fontSize: '0.9rem', color: 'var(--text-primary)' }}>
      {content || 'Not documented.'}
    </div>
  </div>
);

const SOAPPanel = ({ soapNote, isLoading, onGenerate }) => {
  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: '3rem 0' }}>
        <Loader2 className="spin" size={32} color="var(--accent-primary)" />
        <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>Generating SOAP note...</p>
      </div>
    );
  }

  if (!soapNote) {
    return (
      <div style={{ textAlign: 'center', padding: '3rem 0' }}>
        <ClipboardList size={48} color="var(--text-secondary)" style={{ opacity: 0.3, marginBottom: '1rem' }} />
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>Generate a structured clinical SOAP note from the transcript.</p>
        <button className="btn btn-primary" onClick={onGenerate}>
          <ClipboardList size={18} /> Generate SOAP Note
        </button>
      </div>
    );
  }

  return (
    <div className="fade-in">
      <SOAPSection title="Subjective" letter="S" content={soapNote.subjective} color="#3b82f6" />
      <SOAPSection title="Objective" letter="O" content={soapNote.objective} color="#10b981" />
      <SOAPSection title="Assessment" letter="A" content={soapNote.assessment} color="#f59e0b" />
      <SOAPSection title="Plan" letter="P" content={soapNote.plan} color="#8b5cf6" />
    </div>
  );
};

const App = () => {
  const [appState, setAppState] = useState('idle');
  const [isPaused, setIsPaused] = useState(false);
  const [timer, setTimer] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState(null);
  const [recordingBlob, setRecordingBlob] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('transcript');
  const [entities, setEntities] = useState(null);
  const [soapNote, setSoapNote] = useState(null);
  const [isGeneratingSoap, setIsGeneratingSoap] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (appState === 'recording' && !isPaused) {
      timerRef.current = setInterval(() => {
        setTimer((prev) => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [appState, isPaused]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const rawBlob = new Blob(chunksRef.current, { type: mediaRecorderRef.current.mimeType || 'audio/webm' });
        try {
          const wavBlob = await convertRecordingToWav(rawBlob);
          setRecordingBlob(wavBlob);
          processAudio(wavBlob, 'recording.wav');
        } catch (err) {
          setError(err.message || 'Failed to prepare recording for upload.');
          setAppState('idle');
          console.error(err);
        }
      };

      mediaRecorderRef.current.start();
      setAppState('recording');
      setIsPaused(false);
      setTimer(0);
      setError(null);
    } catch (err) {
      setError('Microphone access denied. Please allow microphone permissions.');
      console.error(err);
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.pause();
      setIsPaused(true);
    }
  };

  const resumeRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'paused') {
      mediaRecorderRef.current.resume();
      setIsPaused(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const handleFileUpload = (file) => {
    if (!file) return;
    setUploadedFileName(file.name);
    setError(null);
    processAudio(file, file.name);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFileUpload(file);
    e.target.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileUpload(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const processAudio = async (blob, filename) => {
    setAppState('processing');
    const formData = new FormData();
    formData.append('file', blob, filename || 'audio.wav');

    try {
      const response = await axios.post(`${API_BASE_URL}/process-audio`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setTranscript(response.data.transcript || '');
      setEntities(response.data.entities || null);
      setSoapNote(null);
      setActiveTab('transcript');
      setAppState('result');
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(detail ? `Processing failed: ${detail}` : 'Failed to process audio. Please try again.');
      setAppState('idle');
      console.error(err);
    }
  };

  const handleSave = async () => {
    const editedTranscript = transcript.trim();
    if (!editedTranscript) {
      setError('Transcript cannot be empty.');
      return;
    }

    setIsSaving(true);
    setError(null);
    try {
      await axios.post(`${API_BASE_URL}/save-transcript`, {
        transcript: editedTranscript,
      });
      setTranscript(editedTranscript);
      setAppState('saved');
      setSoapNote(null);

      try {
        const entRes = await axios.get(`${API_BASE_URL}/entities`);
        setEntities(entRes.data.entities || null);
      } catch (_) {}

      setTimeout(() => setAppState('result'), 3000);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(detail ? `Failed to save: ${detail}` : 'Failed to save transcript.');
      console.error(err);
    } finally {
      setIsSaving(false);
    }
  };

  const handleGenerateSoap = async () => {
    setIsGeneratingSoap(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/soap-note`);
      setSoapNote(response.data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(detail ? `SOAP generation failed: ${detail}` : 'Failed to generate SOAP note.');
      console.error(err);
    } finally {
      setIsGeneratingSoap(false);
    }
  };

  const reset = () => {
    setAppState('idle');
    setTranscript('');
    setTimer(0);
    setError(null);
    setRecordingBlob(null);
    setEntities(null);
    setSoapNote(null);
    setActiveTab('transcript');
    setUploadedFileName(null);
  };

  const TabButton = ({ id, label, icon: Icon, count }) => (
    <button
      onClick={() => setActiveTab(id)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.4rem',
        padding: '0.7rem 1.2rem',
        border: 'none',
        borderBottom: activeTab === id ? '2px solid var(--accent-primary)' : '2px solid transparent',
        background: activeTab === id ? 'rgba(6, 182, 212, 0.05)' : 'none',
        color: activeTab === id ? 'var(--accent-primary)' : 'var(--text-secondary)',
        fontWeight: activeTab === id ? '600' : '400',
        cursor: 'pointer',
        fontSize: '0.9rem',
        transition: 'all 0.2s',
        borderRadius: '8px 8px 0 0',
      }}
    >
      <Icon size={16} />
      {label}
      {count > 0 && (
        <span style={{ background: 'var(--accent-primary)', color: '#000', padding: '0.05rem 0.4rem', borderRadius: '999px', fontSize: '0.7rem', fontWeight: '700' }}>{count}</span>
      )}
    </button>
  );

  const entityCount = entities ? Object.values(entities).reduce((sum, arr) => sum + (arr?.length || 0), 0) : 0;

  return (
    <div className="container fade-in">
      <header style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <Stethoscope size={36} color="var(--accent-primary)" />
          <h1>MedScribe</h1>
        </div>
        <p className="subtitle">Clinical Transcription &middot; Medical NER &middot; SOAP Notes &middot; RAG</p>
      </header>

      <main className="glass-card">
        {error && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--error)', marginBottom: '1.5rem', background: 'rgba(239, 68, 68, 0.08)', padding: '0.9rem 1.2rem', borderRadius: '12px', border: '1px solid rgba(239, 68, 68, 0.15)' }}>
            <AlertCircle size={18} />
            <span style={{ flex: 1, fontSize: '0.9rem' }}>{error}</span>
            <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', padding: '0.2rem' }}>✕</button>
          </div>
        )}

        {appState === 'idle' && (
          <div style={{ padding: '2rem 0' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', maxWidth: '600px', margin: '0 auto' }}>
              {/* Record Card */}
              <div
                style={{
                  background: 'var(--bg-secondary)',
                  borderRadius: '16px',
                  padding: '2rem 1.5rem',
                  textAlign: 'center',
                  cursor: 'pointer',
                  border: '1px solid var(--glass-border)',
                  transition: 'all 0.3s',
                }}
                onClick={startRecording}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-primary)'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--glass-border)'; e.currentTarget.style.transform = 'none'; }}
              >
                <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(6, 182, 212, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem' }}>
                  <Mic size={28} color="var(--accent-primary)" />
                </div>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '0.4rem' }}>Record Audio</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Use your microphone to record a consultation</p>
              </div>

              {/* Upload Card */}
              <div
                style={{
                  background: isDragOver ? 'rgba(6, 182, 212, 0.05)' : 'var(--bg-secondary)',
                  borderRadius: '16px',
                  padding: '2rem 1.5rem',
                  textAlign: 'center',
                  cursor: 'pointer',
                  border: isDragOver ? '2px dashed var(--accent-primary)' : '1px solid var(--glass-border)',
                  transition: 'all 0.3s',
                }}
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onMouseEnter={(e) => { if (!isDragOver) { e.currentTarget.style.borderColor = 'var(--accent-primary)'; e.currentTarget.style.transform = 'translateY(-2px)'; }}}
                onMouseLeave={(e) => { if (!isDragOver) { e.currentTarget.style.borderColor = 'var(--glass-border)'; e.currentTarget.style.transform = 'none'; }}}
              >
                <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(139, 92, 246, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem' }}>
                  <Upload size={28} color="#8b5cf6" />
                </div>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '0.4rem' }}>Upload Audio</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Drag & drop or click to upload a file</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_AUDIO_TYPES}
                  onChange={handleFileInputChange}
                  style={{ display: 'none' }}
                />
              </div>
            </div>

            <p style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '1.5rem' }}>
              Supported: WAV, MP3, M4A, FLAC, AAC, OGG, WebM
            </p>
          </div>
        )}

        {appState === 'recording' && (
          <div style={{ textAlign: 'center', padding: '3rem 0' }}>
            <div style={{ marginBottom: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.8rem' }}>
              <div className="pulse" />
              <span style={{ fontSize: '3rem', fontWeight: '700', fontFamily: 'monospace', letterSpacing: '0.05em' }}>{formatTime(timer)}</span>
              <span style={{ color: isPaused ? 'var(--text-secondary)' : 'var(--error)', fontWeight: '600', letterSpacing: '0.1em', fontSize: '0.85rem' }}>
                {isPaused ? 'PAUSED' : 'RECORDING'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
              {isPaused ? (
                <button className="btn btn-secondary" onClick={resumeRecording}>
                  <Play size={20} /> Resume
                </button>
              ) : (
                <button className="btn btn-secondary" onClick={pauseRecording}>
                  <Pause size={20} /> Pause
                </button>
              )}
              <button className="btn btn-primary" style={{ background: 'var(--error)', color: '#fff' }} onClick={stopRecording}>
                <Square size={20} /> Stop & Process
              </button>
            </div>
          </div>
        )}

        {appState === 'processing' && (
          <div style={{ textAlign: 'center', padding: '4rem 0' }}>
            <div className="loader" style={{ marginBottom: '1.5rem' }} />
            <h2 style={{ marginBottom: '0.5rem' }}>Processing{uploadedFileName ? ` "${uploadedFileName}"` : ' Consultation'}</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Transcribing audio, extracting medical entities, generating summary...</p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                <Activity size={14} /> Whisper STT
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                <Tag size={14} /> Medical NER
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                <Stethoscope size={14} /> RAG Analysis
              </div>
            </div>
          </div>
        )}

        {(appState === 'result' || appState === 'saved') && (
          <div className="editor-container fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <CheckCircle2 color="var(--success)" size={22} />
                <h3 style={{ fontSize: '1.3rem' }}>Consultation Analysis</h3>
              </div>
              <button className="btn btn-secondary" onClick={reset} style={{ fontSize: '0.85rem', padding: '0.6rem 1rem' }}>
                <RotateCcw size={16} /> New
              </button>
            </div>

            <div style={{ display: 'flex', borderBottom: '1px solid var(--glass-border)', marginBottom: '1.5rem', gap: '0.25rem' }}>
              <TabButton id="transcript" label="Transcript" icon={FileText} />
              <TabButton id="entities" label="Entities" icon={Tag} count={entityCount} />
              <TabButton id="soap" label="SOAP Note" icon={ClipboardList} />
            </div>

            {activeTab === 'transcript' && (
              <div className="fade-in">
                <textarea
                  className="editor"
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  placeholder="Transcription content will appear here..."
                />
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem', alignItems: 'center' }}>
                  {appState === 'saved' && (
                    <span className="fade-in" style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.3rem', fontWeight: '600', fontSize: '0.9rem' }}>
                      <CheckCircle2 size={16} /> Saved
                    </span>
                  )}
                  <button
                    className="btn btn-primary"
                    onClick={handleSave}
                    disabled={appState === 'saved' || isSaving || !transcript.trim()}
                  >
                    {isSaving ? <Loader2 className="spin" size={18} /> : <Save size={18} />}
                    {isSaving ? 'Saving...' : 'Confirm & Save'}
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'entities' && (
              <div className="fade-in" style={{ minHeight: '300px' }}>
                <EntitiesPanel entities={entities} />
              </div>
            )}

            {activeTab === 'soap' && (
              <div className="fade-in" style={{ minHeight: '300px' }}>
                <SOAPPanel
                  soapNote={soapNote}
                  isLoading={isGeneratingSoap}
                  onGenerate={handleGenerateSoap}
                />
              </div>
            )}
          </div>
        )}
      </main>

      <footer style={{ marginTop: 'auto', padding: '1.5rem 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
        MedScribe &middot; Clinical Documentation Intelligence
      </footer>
    </div>
  );
};

export default App;
