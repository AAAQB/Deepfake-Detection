"use client";

import {
  Camera,
  CircleStop,
  LoaderCircle,
  Play,
  RefreshCw,
  ScanFace,
  ShieldCheck,
  Video,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const SCAN_INTERVAL_MS = 1800;

type Box = { x: number; y: number; width: number; height: number };
type FaceResult = {
  class: string;
  confidence: number;
  probabilities: Record<string, number>;
  box: Box;
};
type PredictionResponse = {
  faces: FaceResult[];
  image: { width: number; height: number };
  inference_ms: number;
};
type Health = {
  status: string;
  error?: string;
  checkpoint?: string;
  device?: string;
  detector?: string;
  input_size?: number;
};
type Activity = {
  id: number;
  time: string;
  label: string;
  confidence: number;
};

function titleCase(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function classTone(label?: string) {
  if (label === "real") return "real";
  if (label === "filter") return "filter";
  return "deepfake";
}

export default function WebcamConsole() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const requestActiveRef = useRef(false);
  const [health, setHealth] = useState<Health>({ status: "checking" });
  const [cameraActive, setCameraActive] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [autoScan, setAutoScan] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [message, setMessage] = useState("Start the camera to analyze a frame.");

  const checkHealth = useCallback(async () => {
    setHealth({ status: "checking" });
    try {
      const response = await fetch(`${API_URL}/health`, { cache: "no-store" });
      const result = (await response.json()) as Health;
      setHealth(result);
    } catch {
      setHealth({
        status: "offline",
        error: `Cannot reach the Python API at ${API_URL}.`,
      });
    }
  }, []);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setAutoScan(false);
    setPrediction(null);
    setMessage("Camera stopped.");
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setMessage("This browser does not support webcam access.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setMessage("Camera ready. Analyze one frame or enable auto scan.");
    } catch {
      setMessage("Camera permission was denied or no webcam is available.");
    }
  };

  const analyzeFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !cameraActive || requestActiveRef.current) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setMessage("Waiting for the webcam image.");
      return;
    }
    requestActiveRef.current = true;
    setAnalyzing(true);
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    try {
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (result) => (result ? resolve(result) : reject(new Error("Capture failed"))),
          "image/jpeg",
          0.86,
        ),
      );
      const form = new FormData();
      form.append("image", blob, "webcam.jpg");
      const response = await fetch(`${API_URL}/predict-frame`, {
        method: "POST",
        body: form,
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail ?? "Inference request failed.");
      }
      const next = result as PredictionResponse;
      setPrediction(next);
      const face = next.faces[0];
      if (face) {
        setMessage(`Detected ${titleCase(face.class)} in ${next.inference_ms} ms.`);
        setActivity((items) =>
          [
            {
              id: Date.now(),
              time: new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              }),
              label: face.class,
              confidence: face.confidence,
            },
            ...items,
          ].slice(0, 6),
        );
      } else {
        setMessage("No face was detected in this frame.");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Inference failed.");
    } finally {
      requestActiveRef.current = false;
      setAnalyzing(false);
    }
  }, [cameraActive]);

  useEffect(() => {
    if (!autoScan || !cameraActive) return;
    void analyzeFrame();
    const timer = window.setInterval(() => void analyzeFrame(), SCAN_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [analyzeFrame, autoScan, cameraActive]);

  const face = prediction?.faces[0];
  const box = face?.box;
  const boxStyle =
    box && prediction
      ? {
          left: `${((prediction.image.width - box.x - box.width) / prediction.image.width) * 100}%`,
          top: `${(box.y / prediction.image.height) * 100}%`,
          width: `${(box.width / prediction.image.width) * 100}%`,
          height: `${(box.height / prediction.image.height) * 100}%`,
        }
      : undefined;

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <ScanFace size={24} aria-hidden="true" />
          <span>Deepfake Lab</span>
          <span className="dev-pill">DEV</span>
        </div>
        <div className="connection">
          <span className={`status-dot ${health.status}`} />
          <span>{health.status === "ready" ? "Model ready" : health.status}</span>
          <span className="separator">•</span>
          <span className="api-address">{API_URL}</span>
        </div>
      </header>

      <div className="workspace">
        <section className="camera-column" aria-label="Webcam controls">
          <div className="camera-stage">
            <video ref={videoRef} playsInline muted />
            {!cameraActive && (
              <div className="camera-empty">
                <div className="camera-icon"><Camera size={34} /></div>
                <h1>Webcam analysis</h1>
                <p>Open your laptop camera and send snapshots to the local frame model.</p>
              </div>
            )}
            {boxStyle && cameraActive && (
              <div className={`face-box ${classTone(face?.class)}`} style={boxStyle}>
                <span>
                  {titleCase(face!.class)} {(face!.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
            <div className="stage-label">
              <span className={`live-dot ${cameraActive ? "on" : ""}`} />
              {cameraActive ? "LIVE CAMERA" : "CAMERA OFF"}
            </div>
          </div>

          <div className="controls">
            <button
              className={cameraActive ? "button secondary" : "button primary"}
              onClick={cameraActive ? stopCamera : startCamera}
            >
              {cameraActive ? <CircleStop size={19} /> : <Play size={19} fill="currentColor" />}
              {cameraActive ? "Stop camera" : "Start camera"}
            </button>
            <button
              className="button secondary"
              disabled={!cameraActive || analyzing || health.status !== "ready"}
              onClick={() => void analyzeFrame()}
            >
              {analyzing ? <LoaderCircle className="spin" size={19} /> : <Video size={19} />}
              {analyzing ? "Analyzing…" : "Analyze frame"}
            </button>
            <button
              className={`button secondary ${autoScan ? "selected" : ""}`}
              disabled={!cameraActive || health.status !== "ready"}
              onClick={() => setAutoScan((value) => !value)}
            >
              <RefreshCw className={autoScan ? "spin-slow" : ""} size={19} />
              {autoScan ? "Stop auto scan" : "Auto scan"}
            </button>
          </div>

          <div className="local-note">
            <ShieldCheck size={17} />
            <span>{message}</span>
            <span className="note-detail">Frames stay on this machine.</span>
          </div>
          <canvas ref={canvasRef} hidden />
        </section>

        <aside className="results-panel" aria-label="Model results">
          <section>
            <div className="section-heading">
              <span>MODEL STATE</span>
              {health.status !== "ready" && (
                <button className="icon-button" onClick={() => void checkHealth()} aria-label="Retry API">
                  <RefreshCw size={16} />
                </button>
              )}
            </div>
            <dl className="model-grid">
              <div><dt>Frame model</dt><dd className={health.status === "ready" ? "ok" : "warning"}>{health.status}</dd></div>
              <div><dt>Weights</dt><dd title={health.checkpoint}>{health.checkpoint?.split("/").pop() ?? "—"}</dd></div>
              <div><dt>Device</dt><dd>{health.device?.toUpperCase() ?? "—"}</dd></div>
              <div><dt>Detector</dt><dd>{health.detector ?? "—"}</dd></div>
              <div><dt>Input size</dt><dd>{health.input_size ? `${health.input_size}×${health.input_size}` : "—"}</dd></div>
            </dl>
            {health.error && <p className="api-error">{health.error}</p>}
          </section>

          <section className="prediction-section">
            <div className="section-heading"><span>PREDICTION</span></div>
            {face ? (
              <>
                <div className={`prediction-name ${classTone(face.class)}`}>
                  {titleCase(face.class)}
                </div>
                <p className="confidence">
                  Confidence <strong>{(face.confidence * 100).toFixed(1)}%</strong>
                  <span>{prediction?.inference_ms} ms</span>
                </p>
              </>
            ) : (
              <div className="empty-result">
                <ScanFace size={28} />
                <p>No analyzed frame yet</p>
              </div>
            )}
          </section>

          <section>
            <div className="section-heading"><span>PROBABILITIES</span></div>
            <div className="probabilities">
              {["real", "filter", "deepfake"].map((label) => {
                const value = face?.probabilities[label] ?? 0;
                return (
                  <div className="probability" key={label}>
                    <div className="probability-label">
                      <span>{titleCase(label)}</span>
                      <span>{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="track">
                      <div
                        className={`fill ${classTone(label)}`}
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="activity-section">
            <div className="section-heading"><span>RECENT ACTIVITY</span></div>
            {activity.length ? (
              <ol className="activity">
                {activity.map((item) => (
                  <li key={item.id}>
                    <time>{item.time}</time>
                    <span className={classTone(item.label)}>{titleCase(item.label)}</span>
                    <strong>{(item.confidence * 100).toFixed(1)}%</strong>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="activity-empty">Predictions from this session will appear here.</p>
            )}
          </section>
        </aside>
      </div>
    </main>
  );
}
