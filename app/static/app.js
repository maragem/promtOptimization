/* Frontend logic: 4-step wizard (dataset -> prompt -> initialise -> ask). */

const $ = (id) => document.getElementById(id);

const STEPS = ["dataset", "prompt", "init", "ask"];
const state = { dataset: null, prompt: "baseline", config: null };

const DS_DESCRIPTIONS = {
  nimbus: "Synthetic helpdesk knowledge base with hallucination-trap questions. Label-free: judged on relevancy + groundedness only.",
  golden: "rag-mini-wikipedia: single-hop factoid questions with gold reference answers. Adds the correctness metric.",
  hotpotqa: "Multi-hop questions needing facts from two passages. Deliberately hard for the naive top-3 retrieval.",
};

// --- Plumbing ----------------------------------------------------------------

async function fetchJSON(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(
        `The request timed out after ${Math.round(timeoutMs / 1000)}s. ` +
        "Check the server logs and try GET /api/test-model for a Bedrock connectivity diagnosis."
      );
    }
    if (err instanceof TypeError) {
      throw new Error(
        "Connection to the server was lost. The server may still be working — " +
        "wait a moment and reload the page, or check the deployment logs."
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
  if (res.status === 401) { window.location.href = "/login.html"; return {}; }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || body.error || `Request failed (${res.status})`);
  return body;
}

function postJSON(url, payload, timeoutMs) {
  return fetchJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, timeoutMs);
}

function spinner(show, text) {
  if (text) $("spinner-text").textContent = text;
  $("spinner").hidden = !show;
}

// --- Stepper -------------------------------------------------------------------

function setStep(active) {
  STEPS.forEach((name, i) => {
    const el = $(`step-${name}`);
    el.classList.toggle("step--active", i === active);
    el.classList.toggle("step--done", i < active);
    el.classList.toggle("step--locked", i > active);
  });
}

function stepIndexOf(name) { return STEPS.indexOf(name); }

// --- Step 1: dataset -------------------------------------------------------------

function renderDatasetCards(data) {
  $("dataset-cards").replaceChildren(
    ...data.datasets.map((ds) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "ds-card" + (ds.active ? " ds-card--selected" : "");
      const badges = [];
      if (ds.golden) badges.push('<span class="badge badge--gold">gold answers</span>');
      if (ds.id === "nimbus") badges.push('<span class="badge">label-free</span>');
      if (ds.id === "hotpotqa") badges.push('<span class="badge badge--warn">multi-hop</span>');
      badges.push(ds.available
        ? `<span class="badge">${ds.questions} questions</span>`
        : '<span class="badge badge--warn">downloads on first use</span>');
      card.innerHTML =
        `<h3>${ds.label}</h3>` +
        `<p>${DS_DESCRIPTIONS[ds.id] || ""}</p>` +
        `<span class="ds-card__badges">${badges.join("")}</span>`;
      card.addEventListener("click", () => selectDataset(ds));
      return card;
    })
  );
}

function datasetStatusText(job) {
  if (job.stage === "fetching") return "Downloading the dataset from Hugging Face…";
  if (job.stage === "indexing") {
    return job.total
      ? `Indexing passages: ${job.indexed || 0}/${job.total}…`
      : "Building the index…";
  }
  return "Preparing…";
}

async function selectDataset(ds) {
  if (!ds.available && !confirm(
    `First use of "${ds.label}" downloads the dataset from Hugging Face and ` +
    "builds the index. This can take a few minutes. Continue?"
  )) return;

  const hint = $("dataset-hint");
  hint.className = "init-status";
  try {
    await postJSON("/api/dataset", { id: ds.id });
  } catch (err) {
    hint.className = "init-status error";
    hint.textContent = err.message;
    return;
  }

  document.querySelectorAll(".ds-card").forEach((c) => (c.disabled = true));
  hint.textContent = datasetStatusText({ stage: "starting" });

  // Poll the background switch job until it finishes.
  while (true) {
    await new Promise((r) => setTimeout(r, 2000));
    let job;
    try {
      job = await fetchJSON("/api/dataset/status");
    } catch (err) {
      hint.textContent = `${err.message} (still checking…)`;
      continue; // transient network blip — keep polling
    }
    if (job.status === "running") {
      hint.textContent = `${datasetStatusText(job)} (${Math.round(job.elapsed_s)}s)`;
    } else if (job.status === "error") {
      hint.className = "init-status error";
      hint.textContent = job.error;
      break;
    } else {
      hint.className = "init-status ok";
      hint.textContent = "Dataset ready.";
      break;
    }
  }
  document.querySelectorAll(".ds-card").forEach((c) => (c.disabled = false));

  // Refresh state from the server (it reverts on failure).
  const [datasets, cfg] = await Promise.all([
    fetchJSON("/api/datasets"),
    fetchJSON("/api/config"),
  ]);
  renderDatasetCards(datasets);
  state.config = cfg;
  state.dataset = datasets.active;
  if (datasets.active === ds.id) {
    state.prompt = "baseline";
    $("summary-dataset").textContent = cfg.dataset_label;
    renderSampleChips(cfg.sample_questions);
    $("answer-panel").hidden = true;
    await enterPromptStep();
  }
}

// --- Step 2: prompt -------------------------------------------------------------

async function refreshPromptView() {
  state.prompt = document.querySelector('input[name="prompt"]:checked').value;
  try {
    const data = await fetchJSON(`/api/prompt/${state.prompt}`);
    $("prompt-view").textContent = data.text;
  } catch (err) {
    $("prompt-view").textContent = `(${err.message})`;
  }
}

function syncOptimizedAvailability() {
  const available = Boolean(state.config?.prompts?.optimized);
  const opt = $("optimized-option");
  opt.classList.toggle("disabled", !available);
  opt.querySelector("input").disabled = !available;
  opt.title = available ? "" : "Run the GEPA optimisation below to create the optimised prompt for this dataset";
  if (!available) {
    document.querySelector('input[name="prompt"][value="baseline"]').checked = true;
  }
}

async function enterPromptStep() {
  syncOptimizedAvailability();
  await refreshPromptView();
  setStep(stepIndexOf("prompt"));
}

function confirmPrompt() {
  $("summary-prompt").textContent =
    (state.prompt === "optimized" ? "Optimised (GEPA) prompt" : "Baseline prompt") +
    ` — ${$("prompt-view").textContent.slice(0, 90)}…`;
  setStep(stepIndexOf("init"));
}

// --- Step 3: initialise -----------------------------------------------------------

async function initialiseModels() {
  const statusEl = $("init-status");
  $("init-btn").disabled = true;
  statusEl.className = "init-status";
  statusEl.textContent =
    "Initialising — verifying the index and making a test call to the LLM. " +
    "The first run can take a couple of minutes (embedding model download)…";
  try {
    const res = await postJSON("/api/init", {}, 300000);
    statusEl.className = "init-status ok";
    statusEl.textContent =
      `Ready. Index check ${res.index_s}s, model test call ${res.model_s}s.`;
    $("summary-init").textContent = `${res.model} @ ${res.region}`;
    setStep(stepIndexOf("ask"));
    $("summary-ask").textContent = "";
  } catch (err) {
    statusEl.className = "init-status error";
    statusEl.textContent = err.message;
  } finally {
    $("init-btn").disabled = false;
  }
}

// --- Step 4: ask ------------------------------------------------------------------

function renderSampleChips(questions) {
  $("sample-chips").replaceChildren(
    ...questions.slice(0, 15).map((q) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip" +
        (q.note === "not_in_kb" ? " chip--trap" : q.answer ? " chip--golden" : "");
      chip.textContent = q.question;
      chip.title = q.note === "not_in_kb"
        ? "Hallucination trap: not covered by the knowledge base"
        : q.answer
          ? `Gold answer: ${q.answer}`
          : "Answerable from the knowledge base";
      chip.addEventListener("click", () => { $("question").value = q.question; });
      return chip;
    })
  );
}

function scoreChip(label, value) {
  const cls = value >= 0.75 ? "good" : value >= 0.4 ? "mid" : "bad";
  const chip = document.createElement("span");
  chip.className = `score-chip score-chip--${cls}`;
  chip.textContent = `${label}: ${value.toFixed(2)}`;
  return chip;
}

function renderAnswer(data) {
  $("answer-panel").hidden = false;
  $("answer-meta").textContent =
    `Prompt: ${data.prompt} · Model: ${data.model} · ${new Date().toLocaleTimeString()}`;
  $("answer-text").textContent = data.answer;

  const scoresEl = $("scores");
  if (data.scores && !data.scores.error) {
    scoresEl.hidden = false;
    const labels = { relevancy: "Relevancy", groundedness: "Groundedness", correctness: "Correctness" };
    $("score-chips").replaceChildren(
      ...Object.entries(labels)
        .filter(([key]) => data.scores[key] != null)
        .map(([key, label]) => scoreChip(label, data.scores[key]))
    );
    $("score-feedback").textContent = data.scores.feedback;
  } else if (data.scores && data.scores.error) {
    scoresEl.hidden = false;
    $("score-chips").replaceChildren();
    $("score-feedback").textContent = data.scores.error;
  } else {
    scoresEl.hidden = true;
  }

  $("context-docs").replaceChildren(
    ...data.documents.map((doc, i) => {
      const details = document.createElement("details");
      details.className = "doc";
      const summary = document.createElement("summary");
      summary.innerHTML = `<span>[${i + 1}] ${doc.id}</span>` +
        (doc.score != null ? `<span class="doc__score">score ${Number(doc.score).toFixed(3)}</span>` : "");
      const pre = document.createElement("pre");
      pre.textContent = doc.content;
      details.append(summary, pre);
      return details;
    })
  );
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) { $("question").focus(); return; }

  const judge = $("judge-toggle").checked;
  spinner(true, judge ? "Calling the model and the judges…" : "Calling the model…");
  $("ask-btn").disabled = true;
  try {
    const data = await postJSON("/api/ask", { question, prompt: state.prompt, judge }, 180000);
    renderAnswer(data);
  } catch (err) {
    alert(err.message);
  } finally {
    spinner(false);
    $("ask-btn").disabled = false;
  }
}

// --- Prompt optimisation (GEPA) ---------------------------------------------------

function renderOptStatus(job) {
  const statusEl = $("opt-status");
  const btn = $("opt-btn");
  if (job.status === "running") {
    btn.disabled = true;
    statusEl.className = "init-status";
    let text = `Running (${job.budget} budget, dataset: ${job.dataset}) — ` +
      `${job.metric_calls} judge evaluations, ${Math.round(job.elapsed_s / 60)} min elapsed.`;
    if (job.last_scores) {
      text += "\nLatest scores: " + Object.entries(job.last_scores)
        .filter(([k]) => k !== "score")
        .map(([k, v]) => `${k} ${Number(v).toFixed(2)}`)
        .join(" · ");
    }
    statusEl.textContent = text;
  } else if (job.status === "done") {
    btn.disabled = false;
    statusEl.className = "init-status ok";
    statusEl.textContent =
      `Finished after ${job.metric_calls} judge evaluations ` +
      `(${Math.round(job.elapsed_s / 60)} min). The optimised prompt is now available in step 2 — ` +
      "remember to commit it to keep it across deploys.";
    $("opt-result").hidden = false;
    $("opt-prompt").textContent = job.prompt || "";
    if (state.config) {
      state.config.prompts.optimized = true;
      syncOptimizedAvailability();
    }
  } else if (job.status === "error") {
    btn.disabled = false;
    statusEl.className = "init-status error";
    statusEl.textContent = `Optimisation failed after ${job.metric_calls || 0} evaluations: ${job.error}`;
  }
}

async function pollOptStatus() {
  let job;
  try {
    job = await fetchJSON("/api/optimize/status");
  } catch (err) {
    console.error(err);
    return;
  }
  renderOptStatus(job);
  if (job.status === "running") setTimeout(pollOptStatus, 4000);
}

async function startOptimisation() {
  if (!confirm(
    "Start the GEPA optimisation for the current dataset? This makes hundreds " +
    "of model calls and typically takes 15–40 minutes."
  )) return;
  try {
    await postJSON("/api/optimize", { budget: "light" });
  } catch (err) {
    alert(err.message);
    return;
  }
  pollOptStatus();
}

// --- Boot -------------------------------------------------------------------------

async function init() {
  spinner(false);

  // Step headers of completed steps navigate back.
  STEPS.forEach((name, i) => {
    $(`step-${name}`).querySelector(".step__header").addEventListener("click", () => {
      if ($(`step-${name}`).classList.contains("step--done")) setStep(i);
    });
  });

  document.querySelectorAll('input[name="prompt"]').forEach((el) =>
    el.addEventListener("change", refreshPromptView)
  );
  $("prompt-continue").addEventListener("click", confirmPrompt);
  $("init-btn").addEventListener("click", initialiseModels);
  $("ask-btn").addEventListener("click", ask);
  $("opt-btn").addEventListener("click", startOptimisation);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
  });

  try {
    const [datasets, cfg] = await Promise.all([
      fetchJSON("/api/datasets"),
      fetchJSON("/api/config"),
    ]);
    state.config = cfg;
    state.dataset = datasets.active;
    renderDatasetCards(datasets);
    renderSampleChips(cfg.sample_questions);
    $("model-badge").textContent = cfg.production_model;
    $("judge-model-note").textContent = `relevancy · groundedness · correctness via ${cfg.judge_model}`;
  } catch (err) {
    $("model-badge").textContent = "offline";
    console.error(err);
  }

  fetchJSON("/api/health")
    .then((h) => { if (h.build) $("build-badge").textContent = `Build ${h.build}.`; })
    .catch(() => {});

  setStep(0);
  pollOptStatus(); // pick up a run already in progress (e.g. page reload)
}

init();
