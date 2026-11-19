// API layer. Two modes:
//   live  (default): talks to the FastAPI backend under /api
//   demo  (VITE_DEMO=true, used for GitHub Pages): no backend; serves results precomputed by
//         scripts/make_demo_assets.py for the 6 bundled sample scans, and keeps reviews in memory.

export const isDemo = import.meta.env.VITE_DEMO === "true";
const base = import.meta.env.BASE_URL;

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

// ---------- demo mode ----------
let demoAll = null;      // the 6 precomputed studies
let demoAdded = [];      // ids the visitor has "uploaded" so far, newest first
async function demoLoad() {
  if (!demoAll) demoAll = await fetch(`${base}demo/studies.json`).then(json);
  return demoAll;
}
export async function demoSamples() {
  return demoLoad();
}
async function demoAdd(id) {
  const all = await demoLoad();
  const s = all.find((x) => x.id === id);
  if (!s) throw new Error("Demo mode only contains the 6 sample scans.");
  if (!demoAdded.includes(id)) demoAdded = [id, ...demoAdded];
  return s;
}

// ---------- shared ----------
export const imageUrl = (id, kind) => (isDemo ? `${base}demo/${id}/${kind}.png` : `/api/studies/${id}/${kind}`);

export const api = {
  health: () =>
    isDemo ? Promise.resolve({ model_loaded: true, orthanc: "static demo" }) : fetch("/api/health").then(json),

  list: async () => {
    if (!isDemo) return fetch("/api/studies").then(json);
    const all = await demoLoad();
    return demoAdded.map((id) => all.find((s) => s.id === id));
  },

  upload: async (file) => {
    if (!isDemo) {
      const fd = new FormData();
      fd.append("file", file);
      return fetch("/api/studies", { method: "POST", body: fd }).then(json);
    }
    const m = /sample_(\d)/.exec(file.name || "");
    if (!m) throw new Error("This static demo only analyses the 6 sample scans. Use the sample buttons.");
    return demoAdd(Number(m[1]) + 1);
  },

  addSample: (id) => demoAdd(id),

  review: async (id, body) => {
    if (!isDemo)
      return fetch(`/api/studies/${id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(json);
    const all = await demoLoad();
    const s = all.find((x) => x.id === id);
    const final = body.final_label || (body.decision === "agree" ? s.ai_label : body.decision === "disagree" ? (s.ai_label === "nodule" ? "normal" : "nodule") : "");
    Object.assign(s, {
      status: "reviewed", review_decision: body.decision, final_label: final,
      review_notes: body.notes, reviewer: body.reviewer, reviewed_at: new Date().toISOString().slice(0, 19) + "Z",
    });
    return s;
  },
};
