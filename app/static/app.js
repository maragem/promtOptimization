/* Frontend logic for the prompt-optimisation demo. */

const $ = (id) => document.getElementById(id);

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
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
    });
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

  await refreshPromptPreview();
  document.querySelectorAll('input[name="prompt"]').forEach((el) =>
    el.addEventListener("change", refreshPromptPreview)
  );
  $("ask-btn").addEventListener("click", ask);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
  });
}

init();
