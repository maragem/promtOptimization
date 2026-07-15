/* Frontend logic for the prompt-optimisation demo. */

const $ = (id) => document.getElementById(id);

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
    throw err;
  } finally {
    clearTimeout(timer);
  }
  if (res.status === 401) { window.location.href = "/login.html"; return {}; }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || body.error || `Request failed (${res.status})`);
  return body;
}

function setReady(ready) {
  $("ask-btn").disabled = !ready;
  $("ask-btn").title = ready ? "" : "Initialise the assistant first";
  $("init-panel").hidden = ready;
}

async function initialiseAssistant() {
  const statusEl = $("init-status");
  $("init-btn").disabled = true;
  statusEl.className = "init-status";
  statusEl.textContent =
    "Initialising — downloading the embedding model, building the index and " +
    "making a test call to the LLM. This can take a couple of minutes on the first run…";
  try {
    const res = await fetchJSON("/api/init", { method: "POST" }, 300000);
    statusEl.className = "init-status ok";
    statusEl.textContent =
      `Ready. Index built in ${res.index_s}s, model test call in ${res.model_s}s ` +
      `(${res.model}, ${res.region}).`;
    setReady(true);
  } catch (err) {
    statusEl.className = "init-status error";
    statusEl.textContent = err.message;
    $("init-btn").disabled = false;
  }
}

function selectedPrompt() {
  return document.querySelector('input[name="prompt"]:checked').value;
}

async function refreshPromptPreview() {
  try {
    const data = await fetchJSON(`/api/prompt/${selectedPrompt()}`);
    $("prompt-text").textContent = data.text;
  } catch (err) {
    $("prompt-text").textContent = `(${err.message})`;
  }
}

// --- Datasets --------------------------------------------------------------

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

function renderDatasets(data) {
  const toggle = $("dataset-toggle");
  const legend = toggle.querySelector("legend");
  toggle.replaceChildren(legend,
    ...data.datasets.map((ds) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "dataset";
      input.value = ds.id;
      input.checked = ds.active;
      input.addEventListener("change", () => switchDataset(ds));
      const span = document.createElement("span");
      span.textContent = ds.label;
      label.append(input, span);
      return label;
    })
  );
  const active = data.datasets.find((d) => d.active);
  $("dataset-hint").textContent = active
    ? (active.golden
        ? `${active.questions} questions with gold answers — the correctness metric is active. Green chips show their reference answer on hover.`
        : `${active.questions ?? "?"} label-free questions — scoring uses relevancy + groundedness only.`)
    : "";
}

async function refreshDatasets() {
  try {
    renderDatasets(await fetchJSON("/api/datasets"));
  } catch (err) {
    console.error(err);
  }
}

async function switchDataset(ds) {
  if (!ds.available && !confirm(
    `First use of "${ds.label}" downloads the dataset from Hugging Face and ` +
    "builds the index. This can take a few minutes. Continue?"
  )) { await refreshDatasets(); return; }

  $("spinner-text").textContent = ds.available
    ? "Switching dataset and rebuilding the index…"
    : "Downloading the golden dataset and building the index…";
  $("spinner").hidden = false;
  try {
    await fetchJSON("/api/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: ds.id }),
    }, 600000);
    const cfg = await fetchJSON("/api/config");
    renderSampleChips(cfg.sample_questions);
    await refreshDatasets();
    $("answer-panel").hidden = true;
  } catch (err) {
    alert(err.message);
    await refreshDatasets(); // snap the radio back to the server's state
  } finally {
    $("spinner").hidden = true;
  }
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

  const docsEl = $("context-docs");
  docsEl.replaceChildren(
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
  $("answer-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- Prompt optimisation (GEPA) ------------------------------------------

function enableOptimizedToggle() {
  const opt = $("optimized-option");
  opt.classList.remove("disabled");
  opt.querySelector("input").disabled = false;
  opt.title = "";
}

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
      `(${Math.round(job.elapsed_s / 60)} min). The optimised prompt is active — ` +
      "remember to commit prompts/optimized.txt to keep it across deploys.";
    $("opt-result").hidden = false;
    $("opt-prompt").textContent = job.prompt || "";
    enableOptimizedToggle();
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
    "Start the GEPA optimisation? This makes hundreds of model calls and " +
    "typically takes 15–40 minutes."
  )) return;
  try {
    await fetchJSON("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ budget: "light" }),
    });
  } catch (err) {
    alert(err.message);
    return;
  }
  pollOptStatus();
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) { $("question").focus(); return; }

  const judge = $("judge-toggle").checked;
  $("spinner-text").textContent = judge
    ? "Calling the model and the judges…"
    : "Calling the model…";
  $("spinner").hidden = false;
  $("ask-btn").disabled = true;
  try {
    const data = await fetchJSON("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, prompt: selectedPrompt(), judge }),
    }, 180000);
    renderAnswer(data);
  } catch (err) {
    alert(err.message);
  } finally {
    $("spinner").hidden = true;
    $("ask-btn").disabled = false;
  }
}

async function init() {
  $("spinner").hidden = true;
  try {
    const cfg = await fetchJSON("/api/config");
    $("model-badge").textContent = cfg.production_model;
    $("judge-model-note").textContent = `relevancy + groundedness via ${cfg.judge_model}`;

    if (!cfg.prompts.optimized) {
      const opt = $("optimized-option");
      opt.classList.add("disabled");
      opt.querySelector("input").disabled = true;
      opt.title = "Run `python -m optimisation.run_gepa` to generate the optimised prompt";
    }

    renderSampleChips(cfg.sample_questions);
  } catch (err) {
    $("model-badge").textContent = "offline";
    console.error(err);
  }

  try {
    const status = await fetchJSON("/api/status");
    setReady(status.ready);
  } catch (err) {
    console.error(err);
  }

  await refreshPromptPreview();
  document.querySelectorAll('input[name="prompt"]').forEach((el) =>
    el.addEventListener("change", refreshPromptPreview)
  );
  $("init-btn").addEventListener("click", initialiseAssistant);
  $("opt-btn").addEventListener("click", startOptimisation);
  $("ask-btn").addEventListener("click", ask);
  refreshDatasets();
  pollOptStatus(); // pick up a run already in progress (e.g. page reload)
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
  });
}

init();
