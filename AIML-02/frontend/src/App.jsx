import React, { useCallback, useEffect, useState } from "react";
import { api, demoSamples, imageUrl, isDemo } from "./api.js";

const pct = (p) => `${(p * 100).toFixed(1)}%`;

function Badge({ study }) {
  if (study.status === "reviewed") return <span className="badge done">Reviewed</span>;
  return <span className="badge wait">Needs review</span>;
}

function Worklist({ studies, selected, onSelect }) {
  if (!studies.length) return <p className="muted">No studies yet. Upload a DICOM, PNG or JPG.</p>;
  return (
    <ul className="worklist">
      {studies.map((s) => (
        <li key={s.id} className={s.id === selected ? "active" : ""} onClick={() => onSelect(s.id)}>
          <div className="row">
            <strong>#{s.id} {s.patient_id || s.filename}</strong>
            <Badge study={s} />
          </div>
          <div className="row small">
            <span className={s.ai_label === "nodule" ? "flag" : ""}>AI: {s.ai_label}</span>
            <span>{pct(s.ai_prob)}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}

function SamplePicker({ onPick, busy }) {
  const [samples, setSamples] = useState([]);
  useEffect(() => { demoSamples().then(setSamples).catch(() => {}); }, []);
  return (
    <div className="samples">
      <div className="muted small">Try a sample scan</div>
      <div className="chips">
        {samples.map((s) => (
          <button key={s.id} type="button" className="chip" disabled={busy} onClick={() => onPick(s.id)}>
            {s.patient_id}
          </button>
        ))}
      </div>
    </div>
  );
}

function Viewer({ study }) {
  const [opacity, setOpacity] = useState(1);
  const [showHeat, setShowHeat] = useState(true);
  return (
    <div>
      <div className="viewer">
        <img src={imageUrl(study.id, "image")} alt="scan" />
        {showHeat && (
          <img className="overlay" style={{ opacity }} src={imageUrl(study.id, "heatmap")} alt="Grad-CAM heatmap" />
        )}
      </div>
      <div className="controls">
        <label>
          <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Show explanation heatmap
        </label>
        <label>
          Opacity
          <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
        </label>
      </div>
      <p className="muted small">
        Warm colours mark the image regions that pushed the model towards “nodule”. It shows where the model looked, not proof that a finding is there.
      </p>
    </div>
  );
}

function ReviewPanel({ study, onSaved }) {
  const [decision, setDecision] = useState("agree");
  const [finalLabel, setFinalLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [reviewer, setReviewer] = useState(localStorage.getItem("reviewer") || "");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => { setDecision("agree"); setFinalLabel(""); setNotes(""); setErr(""); }, [study.id]);

  async function submit(e) {
    e.preventDefault();
    setBusy(true); setErr("");
    try {
      localStorage.setItem("reviewer", reviewer);
      const body = { decision, notes, reviewer };
      if (finalLabel) body.final_label = finalLabel;
      onSaved(await api.review(study.id, body));
    } catch (x) { setErr(x.message); } finally { setBusy(false); }
  }

  if (study.status === "reviewed")
    return (
      <div className="card">
        <h3>Doctor review</h3>
        <p><strong>{study.reviewer}</strong> {study.review_decision === "agree" ? "agreed with" : study.review_decision === "disagree" ? "overrode" : "was unsure about"} the AI.</p>
        <p>Final label: <strong>{study.final_label || "undecided"}</strong></p>
        {study.review_notes && <p className="notes">{study.review_notes}</p>}
        <p className="muted small">{study.reviewed_at}</p>
      </div>
    );

  return (
    <form className="card" onSubmit={submit}>
      <h3>Doctor review</h3>
      <div className="seg">
        {["agree", "disagree", "uncertain"].map((d) => (
          <label key={d} className={decision === d ? "on" : ""}>
            <input type="radio" name="decision" value={d} checked={decision === d} onChange={() => setDecision(d)} />
            {d[0].toUpperCase() + d.slice(1)}
          </label>
        ))}
      </div>
      {decision !== "agree" && (
        <label className="field">Final label
          <select value={finalLabel} onChange={(e) => setFinalLabel(e.target.value)}>
            <option value="">{decision === "disagree" ? "Opposite of AI" : "Undecided"}</option>
            <option value="normal">normal</option>
            <option value="nodule">nodule</option>
          </select>
        </label>
      )}
      <label className="field">Notes
        <textarea rows="3" value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={2000} />
      </label>
      <label className="field">Reviewer
        <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} required placeholder="Dr. name" />
      </label>
      {err && <p className="error">{err}</p>}
      <button disabled={busy || !reviewer.trim()}>{busy ? "Saving…" : "Sign off"}</button>
    </form>
  );
}

export default function App() {
  const [studies, setStudies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try { setStudies(await api.list()); } catch (e) { setErr(e.message); }
  }, []);

  useEffect(() => { refresh(); api.health().then(setHealth).catch(() => setHealth(null)); }, [refresh]);

  async function onFiles(files) {
    setBusy(true); setErr("");
    try {
      let last;
      for (const f of files) last = await api.upload(f);
      await refresh();
      if (last) setSelected(last.id);
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  async function onSample(id) {
    setBusy(true); setErr("");
    try { await api.addSample(id); await refresh(); setSelected(id); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  const study = studies.find((s) => s.id === selected);
  const saved = (s) => setStudies((all) => all.map((x) => (x.id === s.id ? s : x)));

  return (
    <div className="app">
      <header>
        <h1>AI-assisted imaging review</h1>
        <div className="status">
          {isDemo ? "static demo · precomputed results" : health ? `model ${health.model_loaded ? "loaded" : "MISSING"} · Orthanc ${health.orthanc}` : "backend unreachable"}
        </div>
      </header>
      <p className="banner">
        {isDemo
          ? "Static demo: results for 6 sample scans were precomputed by the real model and are served as files. Nothing is analysed in your browser or uploaded anywhere. "
          : ""}
        Research prototype trained on synthetic images. Not a medical device and not for clinical decisions.
      </p>
      <main>
        <aside>
          <label className={`drop ${busy ? "busy" : ""}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); onFiles([...e.dataTransfer.files]); }}>
            {busy ? "Analysing…" : "Drop scans here or click to upload"}
            <input type="file" multiple accept=".dcm,.dicom,.png,.jpg,.jpeg" hidden onChange={(e) => onFiles([...e.target.files])} />
          </label>
          {isDemo && <SamplePicker onPick={onSample} busy={busy} />}
          {err && <p className="error">{err}</p>}
          <Worklist studies={studies} selected={selected} onSelect={setSelected} />
        </aside>
        <section>
          {!study ? <p className="muted">Select a study to review.</p> : (
            <>
              <div className="summary card">
                <div>
                  <div className="muted small">AI prediction</div>
                  <div className={`pred ${study.ai_label === "nodule" ? "flag" : ""}`}>{study.ai_label}</div>
                </div>
                <div className="bar"><div style={{ width: pct(study.ai_prob) }} /></div>
                <div>{pct(study.ai_prob)} probability of nodule</div>
              </div>
              <div className="split">
                <Viewer study={study} />
                <ReviewPanel study={study} onSaved={saved} />
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
