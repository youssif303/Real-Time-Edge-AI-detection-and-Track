import { useEffect, useMemo, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? "https://edgetrack-api.onrender.com" : "");
const POLL_INTERVAL_MS = 5000;

function Metric({ label, value, detail }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function App() {
  const inputRef = useRef(null);
  const pollRef = useRef(null);
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.25);
  const [modelName, setModelName] = useState("yolov8n.pt");
  const [device, setDevice] = useState("cpu");
  const [gpuAvailable, setGpuAvailable] = useState(true);
  const [maxFps, setMaxFps] = useState(5);
  const [imageSize, setImageSize] = useState(640);
  const [result, setResult] = useState(null);
  // status: "idle" | "uploading" | "queued" | "processing" | "complete" | "error"
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [elapsedSec, setElapsedSec] = useState(0);
  const elapsedRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/health`).then((r) => r.json()).then((payload) => {
      const available = Boolean(payload.capabilities?.cuda);
      setGpuAvailable(available);
      if (!available) setDevice("cpu");
    }).catch(() => {});
    return () => stopPolling();
  }, []);

  const annotatedVideoUrl = result?.annotated_video_url
    ? `${API_URL}${result.annotated_video_url}`
    : null;

  const latest = result?.frames?.at(-1);
  const averageLatency = useMemo(() => {
    if (!result?.frames?.length) return 0;
    return result.frames.reduce((sum, f) => sum + f.latency_ms, 0) / result.frames.length;
  }, [result]);
  const classCounts = useMemo(() => {
    return (result?.frames || []).flatMap((f) => f.detections).reduce((counts, d) => {
      counts[d.class_name] = (counts[d.class_name] || 0) + 1;
      return counts;
    }, {});
  }, [result]);

  function stopPolling() {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (elapsedRef.current) { clearInterval(elapsedRef.current); elapsedRef.current = null; }
  }

  function chooseFile(event) {
    const selected = event.target.files?.[0] || null;
    if (selected && selected.size > 100 * 1024 * 1024) {
      setError("File is too large (max 100 MB). Please trim the clip before uploading.");
      return;
    }
    setFile(selected);
    setResult(null);
    setError("");
  }

  async function processVideo() {
    if (!file) return;
    setStatus("uploading");
    setError("");
    setElapsedSec(0);

    const body = new FormData();
    body.append("video", file);
    body.append("confidence_threshold", String(threshold));
    body.append("model_name", modelName);
    body.append("device", device);
    body.append("max_fps", String(maxFps));
    body.append("image_size", String(imageSize));

    let jobId;
    try {
      const response = await fetch(`${API_URL}/api/v1/process`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Failed to submit video.");
      jobId = payload.job_id;
    } catch (err) {
      setError(err.message);
      setStatus("error");
      return;
    }

    // Start elapsed-seconds ticker
    const startTime = Date.now();
    elapsedRef.current = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    setStatus("queued");

    // Poll for job completion
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/jobs/${jobId}`);
        const job = await res.json();
        if (job.status === "processing") {
          setStatus("processing");
        } else if (job.status === "complete") {
          stopPolling();
          setResult(job.result);
          setStatus("complete");
        } else if (job.status === "error") {
          stopPolling();
          setError(job.error || "Processing failed.");
          setStatus("error");
        }
      } catch {
        // Transient network error — keep polling
      }
    }, POLL_INTERVAL_MS);
  }

  function reset() {
    stopPolling();
    setFile(null);
    setResult(null);
    setError("");
    setStatus("idle");
    setElapsedSec(0);
    if (inputRef.current) inputRef.current.value = "";
  }

  const isProcessing = ["uploading", "queued", "processing"].includes(status);
  const statusLabel = status === "uploading" ? "Uploading…"
    : status === "queued" ? `Queued — waiting for worker… (${elapsedSec}s)`
    : status === "processing" ? `Processing frames… (${elapsedSec}s)`
    : status === "processing" ? "Processing frames…"
    : "Run perception pass ↗";

  return (
    <main className="shell">
      <nav className="topbar">
        <div className="brand"><span className="brand-mark">ET</span><span>EdgeTrack</span></div>
        <span className="status-pill"><i /> LOCAL INFERENCE</span>
      </nav>

      <section className="hero">
        <div>
          <p className="eyebrow">PERCEPTION LAB / MILESTONE 02</p>
          <h1>See the system<br /><em>think in motion.</em></h1>
          <p className="lede">Upload a short clip and inspect the objects, identities, and edge performance behind every frame.</p>
        </div>
        <div className="hero-note"><span>01</span><p>Detection<br />plus tracking</p></div>
      </section>

      <section className="workspace">
        <div className="control-panel">
          <div className="section-heading"><span>01 / INPUT</span><span>MP4 · MOV · AVI</span></div>
          <button className={`dropzone ${file ? "has-file" : ""}`} onClick={() => inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); const dropped = e.dataTransfer.files?.[0]; if (dropped?.type.startsWith("video/")) { if (dropped.size > 100 * 1024 * 1024) { setError("File is too large (max 100 MB)."); return; } setFile(dropped); setResult(null); setError(""); } }}>
            <input ref={inputRef} type="file" accept="video/*" onChange={chooseFile} />
            <span className="upload-icon">↑</span>
            <strong>{file ? file.name : "Drop a video here"}</strong>
            <small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB ready to analyze` : "or browse from your machine"}</small>
          </button>
          <div className="settings-grid">
            <label>Model<select value={modelName} onChange={(e) => setModelName(e.target.value)} disabled={isProcessing}><option value="yolov8n.pt">YOLOv8 nano</option><option value="yolov8s.pt">YOLOv8 small</option></select></label>
            <label>Inference size<select value={imageSize} onChange={(e) => setImageSize(Number(e.target.value))} disabled={isProcessing}><option value="416">416 px · faster</option><option value="640">640 px · balanced</option><option value="960">960 px · detailed</option></select></label>
            <label>Device<select value={device} onChange={(e) => setDevice(e.target.value)} disabled={isProcessing}><option value="cpu">CPU</option><option value="0" disabled={!gpuAvailable}>GPU 0{gpuAvailable ? "" : " (unavailable)"}</option></select></label>
          </div>
          <div className="setting-row"><label htmlFor="max-fps">Processing cap</label><output>{maxFps} FPS</output></div>
          <input id="max-fps" className="range" type="range" min="1" max="30" step="1" value={maxFps} onChange={(e) => setMaxFps(Number(e.target.value))} />
          <div className="range-labels"><span>Faster processing</span><span>More frames</span></div>
          <div className="setting-row"><label htmlFor="confidence">Confidence threshold</label><output>{threshold.toFixed(2)}</output></div>
          <input id="confidence" className="range" type="range" min="0" max="1" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          <div className="range-labels"><span>More detections</span><span>More certain</span></div>
          <button className="primary-button" disabled={!file || isProcessing} onClick={processVideo}>
            {isProcessing ? statusLabel : "Run perception pass ↗"}
          </button>
          {elapsedSec > 30 && isProcessing && <p className="error-message">⏳ Still working — the server may be cold-starting or the video is long. This can take 1–2 minutes on the free tier.</p>}
          {result && <button className="text-button" onClick={reset}>Reset workspace</button>}
          {error && <p className="error-message">{error}</p>}
        </div>

        <div className="results-panel">
          <div className="section-heading"><span>02 / TELEMETRY</span><span>{status === "complete" ? "COMPLETE" : "AWAITING INPUT"}</span></div>
          {!result ? (
            <div className="empty-state"><div className="radar"><span /></div><h2>Your results will land here.</h2><p>Choose a clip to start a local YOLO detection and ByteTrack tracking pass.</p></div>
          ) : (
            <>
              <div className="result-header"><div><p className="eyebrow">PROCESSED VIDEO</p><h2>{result.video_name}</h2></div><span className="complete-mark">✓</span></div>
              {annotatedVideoUrl && <div className="video-preview"><video controls src={annotatedVideoUrl} /></div>}
              <div className="metrics-grid">
                <Metric label="Objects / latest" value={latest?.detections?.length ?? 0} detail="visible detections" />
                <Metric label="Active tracks" value={latest?.active_track_count ?? 0} detail="stable identities" />
                <Metric label="Average latency" value={`${averageLatency.toFixed(1)} ms`} detail="per processed frame" />
                <Metric label="Measured FPS" value={`${(result.measured_fps ?? 0).toFixed(2)}`} detail="end-to-end throughput" />
                <Metric label="Total runtime" value={`${((result.processing_time_ms ?? 0) / 1000).toFixed(1)} s`} detail="wall-clock processing" />
                <Metric label="Frames analyzed" value={result.processed_frame_count} detail={`${result.source_fps.toFixed(1)} source FPS`} />
              </div>
              {(result.measured_fps ?? 0) < result.source_fps && <p className="performance-warning">Below source speed: this CPU baseline is processing {(result.measured_fps ?? 0).toFixed(2)} FPS against {result.source_fps.toFixed(1)} FPS input.</p>}
              <div className="class-summary"><div className="list-title">Object classes <span>all processed frames</span></div>{Object.entries(classCounts).map(([name, count]) => <span className="class-chip" key={name}>{name} <b>{count}</b></span>)}</div>
              <div className="detection-list"><div className="list-title">Latest frame detections <span>{latest?.timestamp_ms.toFixed(0)} ms</span></div>{latest?.detections?.length ? latest.detections.map((d) => <div className="detection-row" key={`${d.track_id}-${d.class_id}`}><span className="track-dot" /><strong>{d.class_name}</strong><span>Track {d.track_id ?? "—"}</span><b>{(d.confidence * 100).toFixed(0)}%</b></div>) : <p className="muted">No objects crossed the confidence threshold in the latest frame.</p>}</div>
            </>
          )}
        </div>
      </section>
      <footer><span>YOLO · BYTETRACK · CPU</span><span>Built for measurable perception</span></footer>
    </main>
  );
}

export default App;
