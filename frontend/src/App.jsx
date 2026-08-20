import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  const [audioURL, setAudioURL] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  // Recording timer
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime((previous) => previous + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }

    return () => clearInterval(timerRef.current);
  }, [isRecording]);

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60)
        .toString()
        .padStart(2, "0");

    const remainingSeconds = (seconds % 60)
        .toString()
        .padStart(2, "0");

    return `${minutes}:${remainingSeconds}`;
  };

  const startRecording = async () => {
    try {
      setError(null);
      setResponse(null);
      setAudioURL(null);
      setRecordingTime(0);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      audioChunksRef.current = [];

      const recorder = new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);

        stream.getTracks().forEach((track) => track.stop());

        await sendAudioToBackend(audioBlob);
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Microphone access failed:", error);
      setError("Please allow microphone access.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();

      formData.append("audio", audioBlob, "recording.webm");

      const res = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }

      const data = await res.json();

      console.log("Backend response:", data);

      setResponse(data);
    } catch (error) {
      console.error("Upload failed:", error);

      setError(
          "Could not connect to the backend. Make sure FastAPI is running."
      );
    } finally {
      setUploading(false);
    }
  };

  return (
      <main className="app">
        <section className="shell">

          {/* Navigation */}
          <nav className="nav">
            <div className="logo">
              RAG IN GOA
            </div>

            <div className="nav-label">
              HH26 · VOICE RAG
            </div>
          </nav>

          {/* Hero */}
          <section className="hero">

            <div className="hero-copy">
              <p className="eyebrow">
                Hacker House Goa · 2026
              </p>

              <h1 className="hero-title">
                <span>SPEAK.</span>
                <span>RETRIEVE.</span>
                <span>KNOW.</span>
              </h1>

              <p className="hero-description">
                Ask a question with your voice.
                Our retrieval pipeline finds the
                relevant knowledge and turns it
                into a grounded answer.
              </p>
            </div>

            {/* Recording Card */}
            <div className="record-card">

              <button
                  className={`mic-button ${
                      isRecording ? "recording" : ""
                  }`}
                  onClick={
                    isRecording
                        ? stopRecording
                        : startRecording
                  }
                  disabled={uploading}
                  aria-label={
                    isRecording
                        ? "Stop recording"
                        : "Start recording"
                  }
              >
                {isRecording ? "⏹" : "🎙️"}
              </button>

              <div className="record-status">
                {uploading
                    ? "Transcribing..."
                    : isRecording
                        ? "Listening..."
                        : "Ready to record"}
              </div>

              <div className="record-hint">
                {isRecording
                    ? "Speak your question"
                    : uploading
                        ? "Turning your voice into text"
                        : "Tap the microphone to begin"}
              </div>

              {isRecording && (
                  <div className="recording-timer">
                    {formatTime(recordingTime)}
                  </div>
              )}

              {audioURL && !isRecording && (
                  <audio
                      className="audio-player"
                      controls
                      src={audioURL}
                  />
              )}

            </div>

          </section>

          {/* Results */}
          {(response || error) && (
              <section className="results">

                {/* Transcript */}
                {response?.transcript && (
                    <div className="result-card">
                      <p className="result-label">
                        You said
                      </p>

                      <p className="result-content">
                        {response.transcript}
                      </p>
                    </div>
                )}

                {/* Answer */}
                <div className="result-card">
                  <p className="result-label">
                    Answer
                  </p>

                  <p className="result-content">
                    {response?.answer ||
                        "Your RAG answer will appear here."}
                  </p>

                  {error && (
                      <p className="error">
                        {error}
                      </p>
                  )}
                </div>

              </section>
          )}

        </section>
      </main>
  );
}

export default App;