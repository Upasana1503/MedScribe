import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Mic, Square, Play, Pause, Save, Loader2, AlertCircle, CheckCircle2, RotateCcw, FileText, Tag, ClipboardList } from 'lucide-react';
import './index.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

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
  medications: { label: 'Medications', color: '#3b82f6' },
  conditions: { label: 'Conditions', color: '#ef4444' },
  symptoms: { label: 'Symptoms', color: '#f59e0b' },
  procedures: { label: 'Procedures', color: '#8b5cf6' },
  vitals: { label: 'Vitals', color: '#10b981' },
  anatomy: { label: 'Anatomy', color: '#ec4899' },
  findings: { label: 'Findings', color: '#6366f1' },
  demographics: { label: 'Demographics', color: '#14b8a6' },
  history: { label: 'History', color: '#f97316' },
  temporal: { label: 'Temporal', color: '#64748b' },
  dosages: { label: 'Dosages', color: '#06b6d4' },
};

const EntityBadge = ({ text, color }) => (
  <span style={{
    display: 'inline-block',
    padding: '0.25rem 0.6rem',
    borderRadius: '999px',
    fontSize: '0.8rem',
    fontWeight: '500',
    background: `${color}20`,
    color: color,
    border: `1px solid ${color}40`,
    margin: '0.2rem',
  }}>
    {text}
  </span>
);

const EntitiesPanel = ({ entities }) => {
  if (!entities || Object.keys(entities).length === 0) {
    return <p style={{ color: 'var(--text-secondary)' }}>No medical entities extracted yet.</p>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {Object.entries(entities).map(([category, items]) => {
        const config = ENTITY_LABELS[category] || { label: category, color: '#64748b' };
        if (!items || items.length === 0) return null;
        return (
          <div key={category}>
            <h4 style={{ fontSize: '0.85rem', fontWeight: '600', color: config.color, marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {config.label}
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.15rem' }}>
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

const SOAPSection = ({ title, content, color }) => (
  <div style={{ marginBottom: '1.2rem' }}>
    <h4 style={{ fontSize: '0.9rem', fontWeight: '700', color: color, marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
      {title}
    </h4>
    <div style={{ padding: '0.8rem 1rem', background: 'var(--glass)', borderRadius: '8px', borderLeft: `3px solid ${color}`, whiteSpace: 'pre-wrap', lineHeight: '1.6', fontSize: '0.9rem' }}>
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
        <ClipboardList size={48} color="var(--text-secondary)" style={{ opacity: 0.5, marginBottom: '1rem' }} />
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>Generate a structured clinical SOAP note from the transcript.</p>
        <button className="btn btn-primary" onClick={onGenerate}>
          <ClipboardList size={18} /> Generate SOAP Note
        </button>
      </div>
    );
  }

  return (
    <div className="fade-in">
      <SOAPSection title="Subjective" content={soapNote.subjective} color="#3b82f6" />
      <SOAPSection title="Objective" content={soapNote.objective} color="#10b981" />
      <SOAPSection title="Assessment" content={soapNote.assessment} color="#f59e0b" />
      <SOAPSection title="Plan" content={soapNote.plan} color="#8b5cf6" />
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

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

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
          processAudio(wavBlob);
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

  const processAudio = async (blob) => {
    setAppState('processing');
    const formData = new FormData();
    formData.append('file', blob, 'recording.wav');

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
      setError(detail ? `Failed to process audio: ${detail}` : 'Failed to process audio. Please try again.');
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
      setError(detail ? `Failed to save transcript: ${detail}` : 'Failed to save transcript. Please try again.');
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
  };

  const TabButton = ({ id, label, icon: Icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.4rem',
        padding: '0.6rem 1.2rem',
        border: 'none',
        borderBottom: activeTab === id ? '2px solid var(--accent-primary)' : '2px solid transparent',
        background: 'none',
        color: activeTab === id ? 'var(--accent-primary)' : 'var(--text-secondary)',
        fontWeight: activeTab === id ? '600' : '400',
        cursor: 'pointer',
        fontSize: '0.9rem',
        transition: 'all 0.2s',
      }}
    >
      <Icon size={16} />
      {label}
    </button>
  );

  return (
    <div className="container fade-in">
      <header>
        <h1>MedScribe PoC</h1>
        <p className="subtitle">Intelligent Medical Transcription & Analysis</p>
      </header>

      <main className="glass-card">
        {error && (
          <div className="error-banner" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--error)', marginBottom: '1.5rem', background: 'rgba(239, 68, 68, 0.1)', padding: '1rem', borderRadius: '12px' }}>
            <AlertCircle size={20} />
            <span>{error}</span>
            <button onClick={() => setError(null)} style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>✕</button>
          </div>
        )}

        {appState === 'idle' && (
          <div style={{ textAlign: 'center', padding: '4rem 0' }}>
            <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'center' }}>
              <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'var(--glass)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Mic size={40} color="var(--accent-primary)" />
              </div>
            </div>
            <h2 style={{ marginBottom: '1rem' }}>Ready to Record</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>Start your consultation. We'll handle the transcription and analysis.</p>
            <button className="btn btn-primary" onClick={startRecording}>
              <Mic size={20} /> Start Recording
            </button>
          </div>
        )}

        {appState === 'recording' && (
          <div style={{ textAlign: 'center', padding: '4rem 0' }}>
            <div style={{ marginBottom: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
              <div className="pulse" />
              <span style={{ fontSize: '3rem', fontWeight: '700', fontFamily: 'monospace' }}>{formatTime(timer)}</span>
              <span style={{ color: 'var(--accent-primary)', fontWeight: '600', letterSpacing: '0.1em' }}>
                {isPaused ? 'RECORDING PAUSED' : 'RECORDING LIVE'}
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
            <div className="loader" style={{ marginBottom: '2rem' }} />
            <h2>Processing Consultation</h2>
            <p style={{ color: 'var(--text-secondary)' }}>Transcribing audio, extracting medical entities, and generating summary...</p>
          </div>
        )}

        {(appState === 'result' || appState === 'saved') && (
          <div className="editor-container fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <CheckCircle2 color="var(--success)" size={24} />
                <h3 style={{ fontSize: '1.5rem' }}>Consultation Analysis</h3>
              </div>
              <button className="btn btn-secondary" onClick={reset}>
                <RotateCcw size={18} /> New Recording
              </button>
            </div>

            <div style={{ display: 'flex', borderBottom: '1px solid var(--glass)', marginBottom: '1.5rem' }}>
              <TabButton id="transcript" label="Transcript" icon={FileText} />
              <TabButton id="entities" label="Entities" icon={Tag} />
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
                    <span className="fade-in" style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.3rem', fontWeight: '600' }}>
                      <CheckCircle2 size={18} /> Saved successfully
                    </span>
                  )}
                  <button
                    className="btn btn-primary"
                    onClick={handleSave}
                    disabled={appState === 'saved' || isSaving || !transcript.trim()}
                  >
                    {isSaving ? <Loader2 className="spin" size={20} /> : <Save size={20} />}
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

      <footer style={{ marginTop: 'auto', padding: '2rem 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        &copy; 2026 MedScribe AI. Professional Proof of Concept.
      </footer>
    </div>
  );
};

export default App;
