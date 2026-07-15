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
    const chips = $("score-chips");
    chips.replaceChildren(
      scoreChip("Relevancy", data.scores.relevancy),
      scoreChip("Groundedness", data.scores.groundedness)
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

    $("sample-chips").replaceChildren(
      ...cfg.sample_questions.map((q) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip" + (q.note === "not_in_kb" ? " chip--trap" : "");
        chip.textContent = q.question;
        chip.title = q.note === "not_in_kb"
          ? "Hallucination trap: not covered by the knowledge base"
          : "Answerable from the knowledge base";
        chip.addEventListener("click", () => { $("question").value = q.question; });
        return chip;
      })
    );
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
  $("ask-btn").addEventListener("click", ask);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
  });
}

init();
