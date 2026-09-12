// Brand Support Ops Reviewer Frontend Logic

let allSamples = [];
let activeSample = null;

document.addEventListener("DOMContentLoaded", () => {
  loadEvalResults();
  loadQueueSamples();
  setupEventListeners();
});

function setupEventListeners() {
  document.getElementById("btn-inspect-custom").addEventListener("click", handleCustomInspect);
  document.getElementById("custom-tweet-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleCustomInspect();
  });

  document.getElementById("filter-intent").addEventListener("change", applyFilters);
  document.getElementById("filter-difficulty").addEventListener("change", applyFilters);
}

async function loadEvalResults() {
  try {
    const res = await fetch("api/eval-results");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("kpi-f1").innerText = data.intent_classification.agent_system.macro_f1.toFixed(3);
    document.getElementById("kpi-hit").innerText = (data.retrieval.hit_rate_at_3 * 100).toFixed(1) + "%";
    document.getElementById("kpi-loss").innerText = data.routing_decisions.agent_system.asymmetric_routing_penalty.toFixed(1);
    document.getElementById("kpi-judge").innerText = (data.llm_judge_quality.accept_rate * 100).toFixed(1) + "%";
    document.getElementById("kpi-missed").innerText = data.routing_decisions.agent_system.false_auto_handle_missed_escalate;
  } catch (err) {
    console.warn("Eval results not loaded yet:", err);
  }
}

async function loadQueueSamples() {
  const queueEl = document.getElementById("queue-list");
  try {
    const res = await fetch("api/samples?limit=200");
    const data = await res.json();
    allSamples = data.samples || [];
    renderQueue(allSamples);

    if (allSamples.length > 0) {
      selectSample(allSamples[0]);
    }
  } catch (err) {
    queueEl.innerHTML = `<div class="queue-error">Failed to load samples: ${err.message}</div>`;
  }
}

function renderQueue(samples) {
  const queueEl = document.getElementById("queue-list");
  if (!samples.length) {
    queueEl.innerHTML = `<div class="queue-empty">No matching samples found.</div>`;
    return;
  }

  queueEl.innerHTML = samples.map((s, idx) => `
    <div class="queue-item ${activeSample && activeSample.id === s.id ? 'active' : ''}" data-id="${s.id}" onclick="onSampleClick('${s.id}')">
      <div class="queue-item-meta">
        <span class="q-id">${s.id}</span>
        <span class="q-badge ${s.difficulty}">${s.difficulty}</span>
      </div>
      <div class="queue-item-intent">${formatIntent(s.gold_intent)}</div>
      <div class="queue-item-snippet">${escapeHtml(s.customer_text)}</div>
    </div>
  `).join("");
}

function applyFilters() {
  const intentFilter = document.getElementById("filter-intent").value;
  const diffFilter = document.getElementById("filter-difficulty").value;

  const filtered = allSamples.filter(s => {
    const matchIntent = (intentFilter === "ALL" || s.gold_intent === intentFilter);
    const matchDiff = (diffFilter === "ALL" || s.difficulty === diffFilter);
    return matchIntent && matchDiff;
  });

  renderQueue(filtered);
}

window.onSampleClick = function(id) {
  const found = allSamples.find(s => s.id === id);
  if (found) {
    selectSample(found);
  }
};

async function selectSample(sample) {
  activeSample = sample;

  // Highlight active in queue
  document.querySelectorAll(".queue-item").forEach(el => {
    el.classList.toggle("active", el.getAttribute("data-id") === sample.id);
  });

  // Populate Customer Card
  document.getElementById("disp-id").innerText = sample.id;
  document.getElementById("disp-difficulty").innerText = sample.difficulty;
  document.getElementById("disp-gold-intent").innerText = sample.gold_intent;
  document.getElementById("disp-gold-route").innerText = sample.gold_route;
  document.getElementById("disp-customer-text").innerText = `"${sample.customer_text}"`;
  document.getElementById("disp-gold-ref").innerText = sample.reference_resolution || "Standard Amazon resolution guidance.";

  // Inspect via API
  await runInspection({
    text: sample.customer_text,
    gold_intent: sample.gold_intent,
    gold_route: sample.gold_route,
    reference_resolution: sample.reference_resolution
  });
}

async function handleCustomInspect() {
  const input = document.getElementById("custom-tweet-input");
  const text = input.value.trim();
  if (!text) return;

  activeSample = null;
  document.querySelectorAll(".queue-item").forEach(el => el.classList.remove("active"));

  document.getElementById("disp-id").innerText = "LIVE_QUERY";
  document.getElementById("disp-difficulty").innerText = "ad-hoc";
  document.getElementById("disp-gold-intent").innerText = "UNLABELED";
  document.getElementById("disp-gold-route").innerText = "UNLABELED";
  document.getElementById("disp-customer-text").innerText = `"${text}"`;
  document.getElementById("disp-gold-ref").innerText = "Ad-hoc live sandbox query; evaluated dynamically.";

  await runInspection({ text });
}

async function runInspection(payload) {
  try {
    const res = await fetch("api/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("API inspection failed");
    const data = await res.json();
    updateInspectorUI(data);
  } catch (err) {
    console.error("Inspection error:", err);
  }
}

function updateInspectorUI(data) {
  // 1. Intent Classifier
  const intent = data.intent;
  document.getElementById("disp-intent-pred").innerText = intent.intent;
  document.getElementById("disp-intent-conf").innerText = `${(intent.confidence * 100).toFixed(1)}% Conf`;

  const distEl = document.getElementById("disp-intent-bars");
  const sortedDist = Object.entries(intent.distribution || {}).sort((a, b) => b[1] - a[1]);
  distEl.innerHTML = sortedDist.slice(0, 4).map(([name, prob]) => `
    <div class="intent-bar-row">
      <span class="intent-bar-name" title="${name}">${formatIntent(name)}</span>
      <div class="intent-bar-track">
        <div class="intent-bar-fill" style="width: ${Math.max(prob * 100, 3)}%"></div>
      </div>
      <span class="intent-bar-pct">${(prob * 100).toFixed(0)}%</span>
    </div>
  `).join("");

  // 2. Routing Engine
  const routing = data.routing;
  const routeBanner = document.getElementById("disp-route-banner");
  const isAuto = (routing.decision === "AUTO_HANDLE");
  routeBanner.className = `route-banner ${isAuto ? 'auto' : 'escalate'}`;
  document.getElementById("disp-route-decision").innerText = routing.decision;
  document.getElementById("disp-route-rule").innerText = routing.rule_triggered;
  document.getElementById("disp-route-reason").innerText = routing.reason;

  // 3. Grounded Retrieval
  const retrieval = data.retrieval;
  document.getElementById("disp-retrieval-support").innerText = `Support: ${retrieval.support_score.toFixed(2)}`;
  const hitsEl = document.getElementById("disp-evidence-list");
  const hits = retrieval.top_hits || [];
  if (!hits.length) {
    hitsEl.innerHTML = `<div class="evidence-empty">No historical pairs retrieved.</div>`;
  } else {
    hitsEl.innerHTML = hits.map((h, i) => `
      <div class="evidence-item">
        <div class="evidence-meta">
          <span>Match #${i + 1} (Tweet: ${h.brand_tweet_id})</span>
          <strong>Sim: ${(h.similarity * 100).toFixed(1)}%</strong>
        </div>
        <div class="evidence-pair">
          <div class="evidence-inquiry"><strong>Inquiry:</strong> "${escapeHtml(h.customer_text)}"</div>
          <div class="evidence-reply"><strong>Resolution:</strong> ${escapeHtml(h.brand_reply)}</div>
        </div>
      </div>
    `).join("");
  }

  // 4. Drafted Reply
  const reply = data.reply;
  document.getElementById("disp-draft-text").innerText = reply.draft;
  document.getElementById("disp-draft-mode").innerText = reply.generation_mode;
  document.getElementById("disp-grounding-ids").innerText = (reply.grounded_on_ids || []).map(id => `#${id}`).join(", ") || "N/A";

  // 5. LLM Judge Audit
  const judge = data.judge;
  const verdictPill = document.getElementById("disp-judge-verdict");
  verdictPill.className = `verdict-pill ${(judge.overall_verdict || 'accept').toLowerCase()}`;
  verdictPill.innerText = judge.overall_verdict || "ACCEPT";

  document.getElementById("disp-score-grounding").innerText = `${judge.grounding_score}/5`;
  document.getElementById("disp-score-correctness").innerText = `${judge.correctness_score}/5`;
  document.getElementById("disp-score-tone").innerText = `${judge.tone_score}/5`;
  document.getElementById("disp-score-action").innerText = `${judge.actionability_score}/5`;
  document.getElementById("disp-judge-note").innerText = judge.audit_rationale || "All rubric checks satisfied.";
}

function formatIntent(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
