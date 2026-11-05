const state = {
  caseInfo: null, settings: null, mode: "interactive", role: null,
  episodes: [], episode: null, session: null, result: null, report: null,
  busy: false, setupBusy: false, settingsBusy: false, episodesLoading: false,
  episodeLoad: 0, traceKind: "exchange", traceIndex: 0, raw: false,
  pendingMessage: null, manualFeedbackLocked: false, operation: null,
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
// Copyable presentation hints only; never insert or submit these automatically.
const requestExamples = {
  distribution_grid_operator: [
    "Peak shaving: please reduce bem_01 demand at bus 3 by 10 kW during 16:00-18:00 America/New_York, on the winter_peak_01 operating day.",
    "Congestion management: request an 8 kW demand reduction from bem_04 at bus 11 during 17:00-19:00 America/New_York, on the congestion_risk_01 operating day.",
    "Renewable absorption: increase bem_09 demand at bus 29 by 6 kW during 12:00-14:00 America/New_York, on the pv_surplus_02 operating day.",
  ],
  building_energy_manager: [
    "Baseline flexibility: assess a 5 kW reduction for bem_02 at bus 5 during 09:00-11:00 America/New_York, on the shoulder_02 operating day.",
    "Market trading: I offer an 8 kW load reduction from bem_07 at bus 21 at 60 USD/MWh during 15:00-17:00 America/New_York, on the summer_peak_02 operating day.",
    "Baseline flexibility: can bem_10 at bus 33 provide 4 kW of demand reduction during 10:00-12:00 America/New_York, on the winter_peak_02 operating day?",
  ],
};
const modeName = (mode) => ({ offline: "Offline", live: "Live API", replay: "Replay" }[mode] || mode || "Offline");
const roleName = (role) => ({ distribution_grid_operator: "DNO", building_energy_manager: "BEM" }[role] || label(role));
const familyName = (family) => ({ renewable_absorption: "Renewable integration", baseline_flexibility: "Flexibility assessment" }[family] || label(family));
const workflowNames = {
  awaiting_physics: "Awaiting physics", needs_clarification: "Needs clarification",
  released: "Exchange completed", blocked: "Stopped · review required",
};
function executionFailure(result) {
  const attempt = result?.physics_attempts?.at(-1);
  if (result?.workflow_status !== "blocked" || attempt?.status !== "failed") return null;
  const errors = String(attempt.error || "").match(/(?:KeyError|ValueError|FileNotFoundError|FileExistsError|RuntimeError|TimeoutExpired|PermissionError):[^\r\n]*/g);
  return {
    reason: errors?.at(-1)?.slice(0, 400) || attempt.error_type || "The physical execution tool failed.",
    phase: /prepare_event_idfs\.py|personal_physics\.py/.test(attempt.error || "") ? "Preparing building simulation inputs" : "Physical validation",
    attempts: result.physics_attempts.length,
  };
}
function workflowLabel(result) {
  return executionFailure(result) ? "Execution failed · stopped" : workflowNames[result?.workflow_status];
}
const actionNames = { message: "Processing request", feedback: "Translating feedback", physics: "Running physical validation", report: "Preparing review report" };
const stageNames = {
  "translator.translate": "Translator · interpreting the request",
  "translator.translate_feedback": "Translator · translating receiver feedback",
  "actor.plan": "Actor · preparing the execution plan",
  "actor.free_plan": "Actor · preparing the execution plan",
  "reviewer.review": "Reviewer · checking the result",
  "retrieval.translator": "Retrieving domain knowledge",
  "retrieval.reviewer_memory": "Retrieving review evidence",
  "llm.structured": "Waiting for the model response",
  "llm.free_text": "Waiting for the model response",
};
const initialTrace = $("#trace-content").innerHTML;

async function boot() {
  setDetails(!window.matchMedia("(max-width: 1279px)").matches);
  try {
    const [caseInfo, settings] = await Promise.all([api("/api/case"), api("/api/ui/settings")]);
    state.caseInfo = caseInfo;
    state.settings = settings;
    $("#bem-locations").innerHTML = caseInfo.bems.map((bem) => '<option value="' + e(bem.bem_id) + '">' + e(bem.bem_id.toUpperCase() + " · Bus " + bem.bus) + '</option>').join("");
    $("#bem-locations").disabled = false;
    const readiness = caseInfo.readiness;
    $("#readiness-dot").classList.add(readiness.ready_for_physics ? "ready" : "warn");
    renderSettings();
    renderControls();
  } catch (error) {
    $("#status-text").textContent = "Connection unavailable";
    showError(error.message + " If the server was already running, restart the local UI to load the interface update.");
  }
}

$$("[data-role]").forEach((button) => button.addEventListener("click", async () => {
  if (locked()) return;
  state.role = button.dataset.role;
  $$("[data-role]").forEach((item) => {
    item.classList.toggle("selected", item === button);
    item.setAttribute("aria-pressed", String(item === button));
  });
  if (!state.session) $("#sender-role").value = state.role;
  await loadEpisodes();
}));
$("#session-mode").addEventListener("change", async () => {
  state.mode = $("#session-mode").value;
  state.episode = null;
  $("#seed-settings").open = state.mode === "benchmark";
  if (state.role) await loadEpisodes();
  else renderControls();
});
$("#family-filter").addEventListener("change", loadEpisodes);
$("#episode-select").addEventListener("change", () => {
  state.episode = state.episodes.find((item) => item.episode_id === $("#episode-select").value) || null;
  renderControls();
});
$("#sender-role").addEventListener("change", renderInputExamples);

function renderInputExamples() {
  const input = $("#message-input");
  const panel = $("#input-examples");
  // Sidebar settings select the next session; Message role selects the active sender.
  const role = state.session ? $("#sender-role").value : state.role;
  const examples = requestExamples[role] || [];
  if (panel.dataset.exampleRole !== (role || "")) {
    panel.dataset.exampleRole = role || "";
    panel.innerHTML = examples.map((text, index) => '<div class="input-example"><p tabindex="0">e.g., ' + e(text) + '</p><button type="button" class="copy-example icon-button" data-example="' + index + '" aria-label="Copy example ' + (index + 1) + '" title="Copy example"><svg class="icon" aria-hidden="true"><use href="#i-copy"/></svg></button></div>').join("");
  }
  const visible = examples.length > 0 && !input.value;
  input.placeholder = examples.length ? "Type your request…" : "Start a conversation to enter your building-grid request…";
  input.rows = visible ? 1 : 3;
  panel.classList.toggle("hidden", !visible);
  $("#message-form").classList.toggle("showing-examples", visible);
  resizeInput();
}

$("#input-examples").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-example]");
  if (!button) return;
  const text = requestExamples[$("#input-examples").dataset.exampleRole]?.[Number(button.dataset.example)];
  if (!text) return;
  const focused = document.activeElement;
  try {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(text);
    } catch {
      const selection = window.getSelection();
      const ranges = selection ? Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange()) : [];
      const copyInput = document.createElement("textarea");
      copyInput.value = text;
      copyInput.className = "clipboard-buffer";
      copyInput.setAttribute("aria-hidden", "true");
      document.body.append(copyInput);
      try {
        copyInput.select();
        if (!document.execCommand("copy")) throw new Error("Clipboard unavailable");
      } finally {
        copyInput.remove();
        focused?.focus({ preventScroll: true });
        if (selection) { selection.removeAllRanges(); ranges.forEach((range) => selection.addRange(range)); }
      }
    }
    toast("Example copied. Paste it into your request to edit.");
  } catch {
    toast("Select the example text and use Ctrl / ⌘ + C to copy.", true);
  }
});

async function loadEpisodes() {
  if (!state.role) return;
  const sequence = ++state.episodeLoad;
  state.episodesLoading = true;
  state.episode = null;
  renderControls();
  try {
    const params = new URLSearchParams({ role: state.role, limit: "200" });
    if ($("#family-filter").value) params.set("family", $("#family-filter").value);
    const payload = await api("/api/episodes?" + params);
    if (sequence !== state.episodeLoad) return;
    state.episodes = payload.episodes;
    const first = state.mode === "interactive" ? "No seed · start free-form" : "Choose a frozen episode";
    $("#episode-select").innerHTML = '<option value="">' + first + "</option>" +
      state.episodes.map((episode) => '<option value="' + e(episode.episode_id) + '">' +
        e(episode.episode_id + " · " + familyName(episode.primary_family) + " · " + episode.operating_day) + "</option>").join("");
  } catch (error) {
    if (sequence === state.episodeLoad) {
      state.episodes = [];
      $("#episode-select").innerHTML = '<option value="">Could not load episodes</option>';
      toast(error.message, true);
    }
  } finally {
    if (sequence === state.episodeLoad) {
      state.episodesLoading = false;
      renderControls();
    }
  }
}

$("#open-session").addEventListener("click", async () => {
  if (locked() || !state.role || (state.mode === "benchmark" && !state.episode)) return;
  state.setupBusy = true;
  renderControls();
  try {
    const body = { role: state.role, mode: state.mode };
    if (state.episode) body.episode_id = state.episode.episode_id;
    state.session = await api("/api/sessions", { method: "POST", body: JSON.stringify(body) });
    state.result = null;
    state.report = null;
    state.pendingMessage = null;
    state.operation = null;
    state.manualFeedbackLocked = false;
    state.traceIndex = 0;
    state.raw = false;
    $("#message-input").value = state.episode ? state.episode.sender_message : "";
    $("#sender-role").value = state.role;
    $("#empty-state").classList.add("hidden");
    $("#messages").classList.remove("hidden");
    $("#session-strip").classList.remove("hidden");
    $("#session-context").classList.remove("hidden");
    $("#progress-card").classList.add("hidden");
    $("#request-error").classList.add("hidden");
    $("#manual-feedback").open = false;
    $(".app-shell").classList.remove("config-open");
    $("#config-toggle").setAttribute("aria-expanded", "false");
    renderContext();
    renderMessages();
    renderTrace();
    renderTurnSelect();
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.setupBusy = false;
    renderSettings();
    renderControls();
    resizeInput();
    if (state.session) $("#message-input").focus();
  }
});

$("#llm-mode").addEventListener("change", async () => {
  if (!state.settings || locked()) return;
  const requested = $("#llm-mode").value;
  state.settingsBusy = true;
  $("#settings-error").classList.add("hidden");
  renderControls();
  $("#llm-setting-state").textContent = "Saving…";
  try {
    state.settings = await api("/api/ui/settings", { method: "PUT", body: JSON.stringify({ llm_mode: requested }) });
    toast("Saved. New conversations will use " + modeName(state.settings.llm_mode) + ".");
  } catch (error) {
    $("#settings-error").textContent = error.message;
    $("#settings-error").classList.remove("hidden");
  } finally {
    state.settingsBusy = false;
    renderSettings();
    renderControls();
  }
});

function renderSettings() {
  if (!state.settings) return;
  $("#llm-mode").value = state.settings.llm_mode;
  $("#model-name").textContent = state.settings.model || "Not configured";
  $("#llm-setting-state").textContent = state.settings.editable ? "Local" : "Managed";
  $$("[data-llm-mode]").forEach((item) => item.classList.toggle("selected", item.dataset.llmMode === state.settings.llm_mode));
  const active = state.session?.ui?.llm_mode || state.settings.llm_mode;
  const ready = state.caseInfo?.readiness?.ready_for_physics;
  $("#status-text").textContent = modeName(active) + " · " + (ready ? "Physics ready" : "Physics pending");
  $("#status-pill").className = "status-pill " + (ready ? "ready" : "warn");
  const sessionMode = state.session?.ui;
  $("#session-llm-mode").textContent = sessionMode ? modeName(sessionMode.llm_mode) + " · " + (sessionMode.model || "Local") : "No active conversation";
}
function locked() { return state.busy || state.setupBusy || state.settingsBusy; }
function renderControls() {
  const disabled = locked();
  const activeMode = state.session?.mode;
  const pending = state.result?.workflow_status === "awaiting_physics";
  $("#message-form").classList.toggle("awaiting-feedback", pending);
  const oneTurnUsed = activeMode === "benchmark" && (state.session?.results?.length || 0) > 0;
  $("#open-session").disabled = disabled || state.episodesLoading || !state.caseInfo || !state.role || (state.mode === "benchmark" && !state.episode);
  $("#open-session span").textContent = state.setupBusy ? "Opening…" : state.session ? "New conversation" : "Start conversation";
  $("#setup-help").textContent = state.episodesLoading ? "Loading task catalog…" : !state.role ? "Choose a sender role to start."
    : state.mode === "benchmark" && !state.episode ? "Select a frozen episode to rehearse."
    : state.session ? "These settings apply to your next conversation." : "Ready to open your workspace.";
  $$("[data-role], #session-mode, #family-filter").forEach((item) => { item.disabled = disabled; });
  $("#episode-select").disabled = disabled || state.episodesLoading || !state.role;
  $("#llm-mode").disabled = disabled || !state.settings?.editable;
  $("#send-message").disabled = disabled || !state.session || pending || oneTurnUsed;
  $("#message-input").disabled = disabled || !state.session || pending || oneTurnUsed;
  $("#sender-role").disabled = disabled || !state.session || activeMode === "benchmark" || pending;
  $("#send-message span").textContent = state.busy && state.operation?.action === "message" ? "Processing…" : "Send request";
  $("#finalize").disabled = disabled || !state.result;
  $("#run-physics").disabled = disabled;
  $("#run-physics span").textContent = state.busy && state.operation?.action === "physics" ? "Running…" : "Run physics";
  $("#feedback-section").classList.toggle("hidden", !pending);
  $$("#feedback-form input, #feedback-form textarea, #feedback-form select, #feedback-form button").forEach((item) => {
    item.disabled = disabled || state.manualFeedbackLocked;
  });
  if (state.session) {
    const benchmark = activeMode === "benchmark";
    $("#interaction-heading").textContent = benchmark ? "One exchange, fully traceable." : "Building-Grid Interaction";
    $("#session-mode-badge").classList.remove("hidden");
    $("#session-mode-badge").textContent = benchmark ? "Benchmark" : "Interactive";
    $("#session-label").textContent = "Session " + state.session.session_id + " · " + state.session.results.length + " turn" + (state.session.results.length === 1 ? "" : "s");
    $("#workflow-status").textContent = state.busy ? "Processing" : workflowLabel(state.result) || "Ready for your request";
    $("#workflow-status").className = "workflow-badge " + (state.busy ? "" : state.result?.workflow_status || "");
  }
  $("#composer-help").textContent = pending ? "Awaiting receiver feedback / physical validation."
    : oneTurnUsed ? "Benchmark exchange submitted. Start a new conversation for another request."
    : "";
  $("#composer-help").classList.toggle("hidden", !pending && !oneTurnUsed);
  const failure = executionFailure(state.result);
  if (failure && !disabled) showError(failure.phase + ": " + failure.reason + " Execution has stopped. No further automatic retry. See Physics details before starting another request.");
  renderInputExamples();
}
function renderContext() {
  const episode = state.session.episode;
  const rows = episode ? [
    ["Day", episode.operating_day], ["BEMs", episode.target_bems.join(", ")],
    ["Buses", episode.target_buses.join(", ")], ["Scope", episode.difficulty.spatial_scope + ", " + episode.difficulty.temporal_scope],
    ["Difficulty", episode.difficulty.communication + ", " + episode.difficulty.feasibility],
  ] : [["Protocol", "Interactive"]];
  $("#episode-meta").innerHTML = definitions(rows);
}

$("#message-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("#message-input").value.trim();
  if (!text || !state.session || $("#send-message").disabled) return;
  state.pendingMessage = { speaker: "sender", role: $("#sender-role").value, text, pending: true, created_at: new Date().toISOString() };
  renderMessages();
  try {
    const payload = await perform("message", { text, sender_role: $("#sender-role").value });
    state.pendingMessage = null;
    state.manualFeedbackLocked = false;
    acceptPayload(payload);
    if (state.session.mode === "interactive" && state.result.workflow_status !== "awaiting_physics") $("#message-input").value = "";
  } catch {
    if (state.pendingMessage) { state.pendingMessage.pending = false; state.pendingMessage.failed = true; }
    await recoverSession();
  } finally {
    renderMessages();
    renderControls();
    resizeInput();
  }
});
$("#feedback-decision").addEventListener("change", () => {
  const decision = $("#feedback-decision").value;
  $("#feedback-completion").value = decision === "accepted" ? "100" : decision === "rejected" ? "0" : "50";
  if (decision === "rejected") $("#feedback-power").value = "";
  $("#feedback-basis").value = decision === "accepted" ? "The receiver accepted every requested term without modification."
    : decision === "rejected" ? "The receiver committed none of the requested service."
    : "Completion is the receiver's delivered-versus-requested service ratio.";
});
$("#feedback-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (locked() || !state.session || state.manualFeedbackLocked) return;
  const text = $("#feedback-text").value.trim();
  if (!text) return;
  const power = $("#feedback-power").value.trim();
  try {
    const payload = await perform("feedback", {
      text, decision: $("#feedback-decision").value,
      completion_percentage: Number($("#feedback-completion").value),
      completion_basis: $("#feedback-basis").value.trim(),
      committed_power_kw: power ? Number(power) : null,
    });
    acceptPayload(payload);
    $("#feedback-text").value = "";
    state.manualFeedbackLocked = state.result.workflow_status === "awaiting_physics";
    if (state.manualFeedbackLocked) {
      $("#manual-feedback").open = false;
      toast("Feedback saved as a proposal. Run local physics before release.");
    } else if (state.session.mode === "interactive") $("#message-input").value = "";
  } catch { await recoverSession(); }
  renderControls();
});
$("#run-physics").addEventListener("click", async () => {
  if (locked() || !state.session) return;
  try {
    acceptPayload(await perform("physics", {}));
    if (state.session.mode === "interactive" && state.result.workflow_status !== "awaiting_physics") $("#message-input").value = "";
  } catch { await recoverSession(); }
  renderControls();
});
$("#finalize").addEventListener("click", async () => {
  if (locked() || !state.session) return;
  try {
    state.report = await perform("report", {});
    $("#final-report").innerHTML = renderMarkdown(state.report.report_markdown);
    $("#report-dialog").showModal();
  } catch { /* The persistent request error contains the failure. */ }
  renderControls();
});
$("#close-report").addEventListener("click", () => $("#report-dialog").close());
$("#download-report").addEventListener("click", () => {
  if (!state.report) return;
  const url = URL.createObjectURL(new Blob([state.report.report_markdown], { type: "text/markdown;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = (state.report.episode_id || state.report.session_id) + "-review.md";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function perform(action, body) {
  const operation = {
    id: crypto.randomUUID ? crypto.randomUUID() : "ui-" + Date.now() + "-" + Math.random().toString(16).slice(2),
    action, sessionId: state.session.session_id, start: performance.now(), events: [], failed: false, finished: false,
  };
  state.operation = operation;
  state.busy = true;
  $("#request-error").classList.add("hidden");
  $("#progress-card").className = "progress-card";
  $("#progress-title").textContent = actionNames[action];
  $("#progress-subtitle").textContent = action === "report" ? "Formatting the existing reviewer output…" : "Submitting to the local workspace…";
  $("#progress-events").innerHTML = "";
  renderProgress(operation);
  renderControls();
  const timer = setInterval(() => {
    if (state.operation === operation && state.busy) $("#progress-elapsed").textContent = elapsed((performance.now() - operation.start) / 1000);
  }, 1000);
  const progressUrl = "/api/sessions/" + operation.sessionId + "/progress?operation_id=" + encodeURIComponent(operation.id);
  const poll = async () => {
    try {
      const progress = await api(progressUrl, { signal: AbortSignal.timeout(4000) });
      if (state.operation !== operation || operation.finished) return;
      if (progress.status !== "waiting") {
        operation.events = progress.events || [];
        renderProgress(operation);
      }
    } catch {
      if (state.operation === operation && !operation.finished) $("#progress-subtitle").textContent = "Live updates unavailable; the request is still processing.";
    }
    if (state.operation === operation && !operation.finished) operation.pollTimer = setTimeout(poll, 700);
  };
  void poll();
  try {
    const endpoint = { message: "messages", feedback: "feedback", physics: "execute", report: "finalize" }[action];
    const payload = await api("/api/sessions/" + operation.sessionId + "/" + endpoint, {
      method: "POST", headers: { "X-UI-Operation-ID": operation.id }, body: JSON.stringify(body),
    });
    operation.finished = true;
    const workflow = payload.result?.workflow_status;
    const failure = executionFailure(payload.result);
    $("#progress-card").className = "progress-card " + (failure ? "failed" : workflow === "blocked" ? "blocked" : "completed");
    $("#progress-title").textContent = workflowLabel(payload.result) || (action === "report" ? "Review report ready" : "Processing completed");
    $("#progress-subtitle").textContent = failure ? failure.phase + " failed after " + failure.attempts + " attempt(s). No further automatic retry."
      : workflow === "awaiting_physics" ? "The agent stage returned. Physical execution starts when you choose Run physics."
      : workflow === "needs_clarification" ? "Review the clarification and supply the missing information."
      : workflow === "blocked" ? "This execution has stopped. Review the findings before starting another request."
      : "The result is ready to inspect.";
    // Fetch the final journal view without delaying display of the result.
    void api(progressUrl, { signal: AbortSignal.timeout(4000) }).then((progress) => {
      if (state.operation === operation) { operation.events = progress.events || []; renderProgress(operation); }
    }).catch(() => {});
    return payload;
  } catch (error) {
    operation.failed = true;
    operation.finished = true;
    $("#progress-card").className = "progress-card failed";
    $("#progress-title").textContent = "Request did not complete";
    $("#progress-subtitle").textContent = "Your input is preserved. No request was automatically retried.";
    showError(error.message + " Your input has been preserved. If the connection was interrupted, check the session state before resending.");
    throw error;
  } finally {
    operation.finished = true;
    clearInterval(timer);
    clearTimeout(operation.pollTimer);
    state.busy = false;
    $("#progress-elapsed").textContent = elapsed((performance.now() - operation.start) / 1000);
    renderProgress(operation);
    renderControls();
  }
}
function renderProgress(operation) {
  const events = operation.events;
  const stageEvents = events.filter((item) => item.stage);
  const spans = new Map();
  stageEvents.forEach((item) => spans.set(item.span_id || item.stage, item));
  const active = [...spans.values()].filter((item) => item.event === "stage.started" && !String(item.stage).startsWith("pipeline.")).pop();
  if (!operation.finished && active) $("#progress-subtitle").textContent = stageNames[active.stage] || label(active.stage.replaceAll(".", " "));
  const names = ["Translator", "Actor", "Reviewer"];
  $("#agent-stages").innerHTML = operation.action === "report" ? "" : names.map((name) => {
    const matching = stageEvents.filter((item) => String(item.stage).startsWith(name.toLowerCase() + "."));
    const latest = matching.at(-1);
    const status = !latest ? "" : latest.event === "stage.completed" ? "done" : latest.event === "stage.failed" ? "failed" : operation.finished ? "" : "active";
    const text = status === "done" ? " · done" : status === "active" ? " · working" : status === "failed" ? " · failed" : "";
    return '<span class="agent-stage ' + status + '">' + name + text + "</span>";
  }).join("");
  $("#progress-events").innerHTML = events.filter((item) => item.event !== "llm.response").slice(-30).map((item) => {
    let title = stageNames[item.stage] || label(String(item.stage || item.event).replaceAll(".", " "));
    if (item.event === "reviewer.return") title = "Reviewer returned to " + label(item.route) + (item.revision ? " · revision " + item.revision : "");
    else if (item.event === "llm.request") title = (item.agent ? label(item.agent) + " · " : "") + "Model request";
    else if (item.event === "llm.format_repair") title = "Response format repair";
    else if (item.event === "stage.completed") title += " · done";
    else if (item.event === "stage.failed") title += " · failed";
    return "<li><span>" + e(title) + "</span><time>" + e(item.elapsed_seconds != null ? Number(item.elapsed_seconds).toFixed(1) + "s" : clockTime(item.timestamp)) + "</time></li>";
  }).join("");
}
async function recoverSession() {
  try {
    const snapshot = await api("/api/sessions/" + state.session.session_id, { signal: AbortSignal.timeout(4000) });
    const pending = state.pendingMessage;
    const last = snapshot.messages.at(-1);
    if (pending && last?.speaker === "sender" && last.text === pending.text && last.role === pending.role) state.pendingMessage = null;
    state.session = snapshot;
    state.result = snapshot.latest_result;
    state.traceIndex = Math.max(0, snapshot.results.length - 1);
    renderTurnSelect();
    renderMessages();
    renderTrace();
  } catch { /* Keep the visible draft when the session cannot be read. */ }
}
function acceptPayload(payload) {
  state.session = payload.snapshot;
  state.result = payload.result;
  state.report = null;
  state.traceIndex = Math.max(0, state.session.results.length - 1);
  renderMessages();
  renderTurnSelect();
  renderTrace();
  renderControls();
}
function renderMessages() {
  if (!state.session) return;
  const messages = [...state.session.messages];
  if (state.pendingMessage) messages.push(state.pendingMessage);
  let request = 0;
  $("#messages").innerHTML = messages.map((message) => {
    const sender = message.speaker === "sender";
    const divider = sender ? '<div class="turn-divider">REQUEST ' + String(++request).padStart(2, "0") + "</div>" : "";
    const names = { sender: roleName(message.role), agent_to_receiver: "Agent → receiver", receiver: "Receiver feedback", agent_to_sender: "Agent → sender", system: "Workspace" };
    const initials = { sender: message.role === "building_energy_manager" ? "BM" : "GO", agent_to_receiver: "AG", receiver: "RF", agent_to_sender: "AG", system: "" };
    return divider + '<article class="message ' + e(message.speaker) + (message.pending ? " pending" : message.failed ? " failed" : "") + '"><span class="message-avatar" aria-hidden="true">' + initials[message.speaker] +
      '</span><div class="message-body"><div class="message-meta"><strong>' + e(names[message.speaker]) + "</strong><span>" + e(clockTime(message.created_at)) +
      "</span></div><p>" + e(message.text) + "</p>" + (message.pending ? '<span class="message-state">Submitting / processing…</span>' : message.failed ? '<span class="message-state">Not confirmed · draft preserved</span>' : "") + "</div></article>";
  }).join("");
  $("#messages").scrollTop = $("#messages").scrollHeight;
}
function renderTurnSelect() {
  const results = state.session?.results || [];
  $("#turn-select").disabled = results.length === 0;
  $("#turn-select").innerHTML = results.length ? results.map((result, index) =>
    '<option value="' + index + '">Turn ' + result.turn_index + " · " + e(workflowLabel(result) || label(result.workflow_status)) + "</option>").join("")
    : '<option value="">No turns yet</option>';
  $("#turn-select").value = String(state.traceIndex);
}
$("#turn-select").addEventListener("change", () => { state.traceIndex = Number($("#turn-select").value); renderTrace(); });
$("#raw-toggle").addEventListener("click", () => { state.raw = !state.raw; renderTrace(); });
$$("[data-trace]").forEach((button) => {
  button.addEventListener("click", () => { state.traceKind = button.dataset.trace; renderTrace(); });
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const tabs = $$("[data-trace]");
    const current = tabs.indexOf(button);
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].click(); tabs[next].focus();
  });
});
function renderTrace() {
  const kind = state.traceKind;
  $$("[data-trace]").forEach((button) => {
    const active = button.dataset.trace === kind;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  $("#raw-toggle").setAttribute("aria-pressed", String(state.raw));
  $("#trace-content").classList.toggle("hidden", state.raw);
  $("#trace-output").classList.toggle("hidden", !state.raw);
  $("#trace-content").setAttribute("aria-labelledby", "tab-" + kind);
  const result = state.session?.results[state.traceIndex] || state.result;
  if (!result) {
    $("#trace-content").innerHTML = initialTrace;
    $("#trace-output").textContent = "Run an interaction to populate the trace.";
    return;
  }
  const value = { exchange: result.information_exchange, task: result.task_specification, plan: result.execution_plan, review: result.reviewer_decision, physics: result.physics_evidence }[kind];
  $("#trace-output").textContent = JSON.stringify(value, null, 2);
  const renderers = { exchange: exchangeView, task: taskView, plan: planView, review: reviewView, physics: physicsView };
  $("#trace-content").innerHTML = renderers[kind](value, result);
}
function exchangeView(exchange, result) {
  const stages = [["Sender sent", "sender_sent"], ["Receiver received", "receiver_received"], ["Receiver sent", "receiver_sent"], ["Sender received", "sender_received"]];
  return card("Workflow", "<p>" + e(workflowLabel(result) || label(exchange.decision)) + "</p>", result.workflow_status === "needs_clarification" ? "warn" : "") +
    stages.map(([name, key], index) => '<section class="detail-card ' + (exchange[key] == null ? "pending" : "") + '"><h3><span class="state-number">' + (index + 1) + "</span>" + name +
      '<span class="state-caption">' + (exchange[key] == null ? "Pending" : "Recorded") + "</span></h3><p>" + e(exchange[key] ?? "Not available at this stage.") + "</p></section>").join("");
}
function taskView(task) {
  const windows = (task.windows || []).map((item) => (item.start || item.start_time || "—") + "–" + (item.end || item.end_time || "—")).join(", ");
  let html = card("Task context", '<dl class="metadata">' + definitions([
    ["Family", task.task_families.map(familyName).join(", ")], ["Sender", roleName(task.sender_role)], ["Receiver", roleName(task.receiver_role)],
    ["Day", task.operating_day || "Not specified"], ["Window", windows || "Not specified"],
    ["BEMs", task.target_bems.join(", ") || "Not specified"], ["Buses", task.target_buses.join(", ") || "Not specified"],
  ]) + "</dl>");
  if (task.clarification_required) html += card("Clarification required", "<p>" + e(task.clarification_question) + "</p>", "warn");
  html += card("Structured fields", Object.entries(task.fields).map(([name, field]) =>
    '<div class="field-row"><div class="field-row-top"><strong>' + e(label(name)) + '</strong><span class="field-state ' + (["missing", "unresolved", "to_be_simulated"].includes(field.status) ? "unresolved" : "") + '">' +
    e(label(field.status)) + "</span></div><p>" + e(display(field.value)) + (field.unit ? " " + e(field.unit) : "") + "</p>" +
    (field.note ? '<p class="small-note">' + e(field.note) + "</p>" : "") + "</div>").join(""));
  if (task.constraints?.length) html += card("Constraints", "<ul>" + task.constraints.map((item) => "<li>" + e(item.name + " · " + label(item.operator) + " " + display(item.value) + " " + item.unit) + "</li>").join("") + "</ul>");
  if (task.requested_outputs?.length) html += card("Requested outputs", "<ul>" + task.requested_outputs.map((item) => "<li>" + e(label(item)) + "</li>").join("") + "</ul>");
  return html;
}
function planView(plan) {
  if (!plan) return emptyDetail("No execution plan yet", "A plan will appear when the request contains the required information.");
  return card(plan.valid ? "Plan compiled" : "Plan requires revision", '<dl class="metadata">' + definitions([["Plan", plan.plan_id], ["Resolution", plan.time_resolution_minutes + " minutes"], ["Actions", plan.actions.length]]) + "</dl>" +
    (plan.validation_errors.length ? "<ul>" + plan.validation_errors.map((item) => "<li>" + e(item) + "</li>").join("") + "</ul>" : ""), plan.valid ? "success" : "warn") +
    plan.actions.map((action, index) => '<div class="plan-step"><span class="step-number">' + (index + 1) + "</span><div><strong>" + e(label(action.action_type)) +
      "</strong><p>" + e(action.tool + " · " + action.action_id) + "</p>" + (action.depends_on.length ? "<p>Depends on " + e(action.depends_on.join(", ")) + "</p>" : "") + "</div></div>").join("");
}
function reviewView(review) {
  let html = card("Reviewer · " + label(review.verdict), "<p>" + e(review.final_message) + '</p><dl class="metadata" style="margin-top:12px">' +
    definitions([["Routing", label(review.routing_decision)], ["Deterministic", yesNo(review.deterministic_checks_passed)], ["Semantic review", label(review.semantic_review_status)], ["Revisions", review.revision_count]]) + "</dl>", review.verdict === "pass" ? "success" : "warn");
  html += review.issues.length ? review.issues.map((issue) => card(issue.code + " · " + label(issue.responsible_agent),
    "<p>" + e(issue.message) + '</p><span class="detail-label" style="margin-top:12px">Required correction</span><p>' + e(issue.correction_required) + "</p>", "warn")).join("")
    : card("Findings", "<p>No review issues were recorded.</p>");
  return html;
}
function physicsView(physics, result) {
  const failure = executionFailure(result);
  if (failure) return card("Execution failed · stopped", '<dl class="metadata">' + definitions([
    ["Stage", failure.phase], ["Reason", failure.reason], ["Attempts", failure.attempts],
    ["Next step", "Inspect the execution error. Sending a confirmation starts a new request and does not resolve a tool failure."],
  ]) + '</dl>', "error");
  if (!physics) return emptyDetail("No physical evidence yet", "Run physical validation to view evidence for this turn.");
  const completion = Number(physics.completion_percentage);
  let html = '<div class="metric-grid"><div class="metric"><strong>' + e(number(physics.committed_power_kw)) + '</strong><span>Committed power · kW</span></div><div class="metric"><strong>' +
    e(number(completion)) + '%</strong><span>Evidence-backed completion</span><progress class="evidence-meter" max="100" value="' + completion + '" aria-label="Validated completion"></progress></div></div>';
  html += card("Physical evidence", '<dl class="metadata">' + definitions([
    ["Status", label(physics.status)], ["Decision", label(physics.receiver_decision)], ["Convergence", yesNo(physics.converged)],
    ["Hard constraints", yesNo(physics.hard_constraints_passed)], ["Workflow", label(physics.workflow)], ["Replans", display(physics.replan_count)],
  ]) + "</dl>", physics.hard_constraints_passed && physics.converged ? "success" : "warn");
  const values = physics.business_metrics?.requested_output_values || {};
  if (Object.keys(values).length) html += card("Requested output values", Object.entries(values).map(([name, value]) =>
    '<div class="field-row"><span class="detail-label">' + e(label(name)) + '</span><span class="detail-value">' + e(display(value)) + "</span></div>").join(""));
  const checks = Object.entries(physics.execution_log || {}).filter(([, value]) => typeof value === "boolean");
  if (checks.length) html += card("Execution checks", '<dl class="metadata">' + definitions(checks.map(([name, value]) => [label(name), yesNo(value)])) + "</dl>");
  return html;
}
function card(title, content, tone = "") { return '<section class="detail-card ' + tone + '"><h3>' + e(title) + "</h3>" + content + "</section>"; }
function emptyDetail(title, text) { return '<div class="detail-empty"><svg class="icon"><use href="#i-layers"/></svg><h3>' + e(title) + "</h3><p>" + e(text) + "</p></div>"; }
function definitions(rows) { return rows.map(([name, value]) => "<dt>" + e(name) + "</dt><dd>" + e(value) + "</dd>").join(""); }
function display(value) {
  if (value == null) return "Not specified";
  if (Array.isArray(value)) {
    if (!value.length) return "None";
    if (value.every((item) => item && typeof item === "object" && item.start && item.end))
      return value.map((item) => item.start + "–" + item.end + (item.timezone ? " · " + item.timezone : "")).join("\n");
    if (value.every((item) => typeof item !== "object")) return value.join(", ");
  }
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
function yesNo(value) { return value === true ? "Passed" : value === false ? "Not passed" : "Not available"; }
function number(value) { return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"; }

function renderMarkdown(markdown) {
  // Render a small, escaped Markdown subset; never execute HTML from reports.
  const tick = String.fromCharCode(96);
  const inline = (text) => e(text).replace(new RegExp(tick + "([^" + tick + "]+)" + tick, "g"), "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const lines = markdown.split(/\r?\n/);
  let html = "", list = "", fenced = false, code = [];
  const closeList = () => { if (list) { html += "</" + list + ">"; list = ""; } };
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (line.startsWith(tick.repeat(3))) {
      closeList();
      if (fenced) { html += "<pre><code>" + e(code.join("\n")) + "</code></pre>"; code = []; }
      fenced = !fenced; continue;
    }
    if (fenced) { code.push(line); continue; }
    if (!line.trim()) { closeList(); continue; }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) { closeList(); const level = heading[1].length; html += "<h" + level + ">" + inline(heading[2]) + "</h" + level + ">"; continue; }
    if (line.trim().startsWith("|") && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[index + 1] || "")) {
      closeList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
      html += "<table><thead><tr>" + cells(line).map((cell) => "<th>" + inline(cell) + "</th>").join("") + "</tr></thead><tbody>";
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        html += "<tr>" + cells(lines[index]).map((cell) => "<td>" + inline(cell) + "</td>").join("") + "</tr>"; index++;
      }
      index--; html += "</tbody></table>"; continue;
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const type = /^\s*\d+\./.test(line) ? "ol" : "ul";
      if (list !== type) { closeList(); html += "<" + type + ">"; list = type; }
      html += "<li>" + inline(line.replace(/^\s*([-*]|\d+\.)\s+/, "")) + "</li>"; continue;
    }
    closeList(); html += "<p>" + inline(line) + "</p>";
  }
  closeList();
  if (code.length) html += "<pre><code>" + e(code.join("\n")) + "</code></pre>";
  return html;
}
function setDetails(open) {
  $(".app-shell").classList.toggle("details-hidden", !open);
  $("#details-toggle").setAttribute("aria-expanded", String(open));
}
window.matchMedia("(max-width: 1279px)").addEventListener("change", (event) => setDetails(!event.matches));
$("#details-toggle").addEventListener("click", () => {
  setDetails($(".app-shell").classList.contains("details-hidden"));
  $(".app-shell").classList.remove("config-open");
  $("#config-toggle").setAttribute("aria-expanded", "false");
});
$("#close-inspector").addEventListener("click", () => { setDetails(false); $("#details-toggle").focus(); });
$("#config-toggle").addEventListener("click", () => {
  const open = !$(".app-shell").classList.contains("config-open");
  $(".app-shell").classList.toggle("config-open", open);
  $("#config-toggle").setAttribute("aria-expanded", String(open));
  if (open) setDetails(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    $(".app-shell").classList.remove("config-open");
    $("#config-toggle").setAttribute("aria-expanded", "false");
    if (window.matchMedia("(max-width: 1279px)").matches) setDetails(false);
  }
});
$("#message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
    event.preventDefault(); if (!$("#send-message").disabled) $("#message-form").requestSubmit();
  }
});
function resizeInput() {
  const input = $("#message-input");
  const style = getComputedStyle(input);
  const limit = parseFloat(style.maxHeight);
  const minimum = parseFloat(style.minHeight);
  input.style.height = "auto";
  input.style.height = Math.min(limit, Math.max(minimum, input.scrollHeight)) + "px";
}
$("#message-input").addEventListener("input", renderInputExamples);
window.addEventListener("resize", resizeInput);

async function api(url, options = {}) {
  let response;
  try { response = await fetch(url, { ...options, headers: { "Content-Type": "application/json", ...options.headers } }); }
  catch (error) { throw new Error(error.name === "TimeoutError" ? "The local status request timed out." : "Could not reach the local workspace."); }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg).join("; ") : "Request failed: " + response.status;
    throw new Error(detail);
  }
  return payload;
}
function e(value) { return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char])); }
function label(value) { return String(value ?? "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function clockTime(value) { if (!value) return ""; const date = new Date(value); return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
function elapsed(seconds) { const value = Math.max(0, Math.floor(seconds)); return Math.floor(value / 60) + ":" + String(value % 60).padStart(2, "0"); }
function showError(message) { $("#request-error").textContent = message; $("#request-error").classList.remove("hidden"); }
let toastTimer;
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").className = "toast show" + (error ? " error" : "");
  toastTimer = setTimeout(() => $("#toast").className = "toast", 5000);
}
void boot();
