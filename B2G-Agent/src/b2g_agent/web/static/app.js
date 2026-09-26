const appState = {
  catalog: null,
  scenario: null,
  role: null,
  session: null,
  latestRun: null,
  finalPlan: null,
};

const scenarioIds = {
  gridUpgrade: "distribution_grid_upgrade",
  demandResponse: "demand_response_service",
};

const roleLabels = {
  building_engineer: "Building Engineer",
  distribution_power_engineer: "Distribution Power Engineer",
};

const parameterLabels = {
  cooling_setpoint_c: ["Cooling setpoint", "°C"],
  building_count: ["Participating buildings", ""],
  retrofit_level: ["Retrofit", ""],
  pv_kw_per_building: ["PV per building", "kW"],
  demand_response_pct: ["Peak flexibility", "%"],
  target_bus: ["Connection", ""],
  transformer_capacity_kva: ["Transformer", "kVA"],
  line_capacity_kw: ["Line capacity", "kW"],
  baseline_method: ["Baseline method", ""],
  baseline_adjustment_pct: ["Baseline adjustment", "%"],
  dr_event_start_hour: ["Event start", ":00"],
  dr_event_duration_hours: ["Event duration", "hours"],
  dr_target_kw_per_building: ["DR target", "kW/building"],
  max_rebound_pct: ["Max rebound", "% of target"],
};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  bindEvents();
  try {
    const [health, catalog] = await Promise.all([
      requestJSON("/api/health"),
      requestJSON("/api/scenarios"),
    ]);
    appState.catalog = catalog;
    document.querySelector("#service-status").textContent = `${health.service} ${health.version} · local service online`;
    document.querySelector("#service-dot").classList.add("online");
    renderScenarioOptions(catalog.scenarios);
  } catch (error) {
    showToast(`Could not connect to the local service: ${error.message}`, true);
    document.querySelector("#service-status").textContent = "Local service unavailable";
  }
}

function bindEvents() {
  document.querySelectorAll(".role-card").forEach((button) => {
    button.addEventListener("click", () => selectRole(button.dataset.role));
  });
  document.querySelector("#accept-brief").addEventListener("change", updateEnterButton);
  document.querySelector("#enter-room").addEventListener("click", enterRoom);
  document.querySelectorAll("[data-back]").forEach((button) => {
    button.addEventListener("click", () => showStage(Number(button.dataset.back)));
  });
  document.querySelector("#composer").addEventListener("submit", sendMessage);
  document.querySelector("#message-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      document.querySelector("#composer").requestSubmit();
    }
  });
  document.querySelector("#rerun-button").addEventListener("click", runCurrentCase);
  document.querySelector("#review-button").addEventListener("click", finalizePlan);
  document.querySelector("#download-report").addEventListener("click", downloadReport);
  document.querySelector("#restart-session-button").addEventListener("click", () => {
    const confirmed = window.confirm(
      "Return to role selection and start a new session? Your current session artifacts will remain saved locally.",
    );
    if (confirmed) resetToStart();
  });
  document.querySelector("#new-session-button").addEventListener("click", resetToStart);
}

function renderScenarioOptions(scenarios) {
  const container = document.querySelector("#scenario-options");
  container.replaceChildren();
  scenarios.forEach((scenario, index) => {
    const button = element("button", "scenario-card");
    button.type = "button";
    button.dataset.scenarioId = scenario.scenario_id;
    const indexNode = element("span", "scenario-index", String(index + 1).padStart(2, "0"));
    const copy = element("span", "scenario-card-copy");
    copy.append(
      element("span", "eyebrow", scenario.kicker),
      element("strong", "", scenario.title),
      element("p", "", scenario.selection_summary),
    );
    button.append(indexNode, copy);
    button.addEventListener("click", () => selectScenario(scenario.scenario_id));
    container.append(button);
  });
}

function selectRole(role) {
  appState.role = role;
  appState.scenario = null;
  document.querySelector("#selected-role-badge").textContent = roleLabels[role];
  document.querySelectorAll(".scenario-card").forEach((card) => card.classList.remove("selected"));
  document.querySelector("#scenario-detail").classList.add("is-hidden");
  document.querySelector("#accept-brief").checked = false;
  updateEnterButton();
  showStage(2);
}

async function selectScenario(scenarioId) {
  try {
    const scenario = await requestJSON(`/api/scenarios/${scenarioId}`);
    appState.scenario = scenario;
    document.querySelectorAll(".scenario-card").forEach((card) => {
      card.classList.toggle("selected", card.dataset.scenarioId === scenarioId);
    });
    populateScenario(scenario);
    document.querySelector("#scenario-detail").classList.remove("is-hidden");
    document.querySelector("#accept-brief").checked = false;
    updateEnterButton();
  } catch (error) {
    showToast(error.message, true);
  }
}

function populateScenario(scenario) {
  const counterpart = appState.role === "building_engineer"
    ? "distribution_power_engineer"
    : "building_engineer";
  const roleInfo = scenario.user_roles.find((item) => item.id === appState.role);
  const counterpartInfo = scenario.user_roles.find((item) => item.id === counterpart);
  document.querySelector("#scenario-kicker").textContent = scenario.kicker;
  document.querySelector("#brief-heading").textContent = scenario.title;
  document.querySelector("#scenario-summary").textContent = scenario.summary;
  document.querySelector("#mission-title").textContent = roleLabels[appState.role];
  document.querySelector("#mission-text").textContent = roleInfo.mission;
  document.querySelector("#counterpart-text").textContent = `${roleLabels[counterpart]} — ${counterpartInfo.mission}`;
  document.querySelector("#simulation-note").textContent = scenario.simulation_note;
  document.querySelector("#room-heading").textContent = scenario.room_title;
  document.querySelector("#scenario-illustration").classList.toggle(
    "demand-response",
    scenario.scenario_id === scenarioIds.demandResponse,
  );
  replaceList("#constraint-list", scenario.hard_constraints);
  const highlights = document.querySelector("#baseline-row");
  highlights.replaceChildren();
  scenario.highlights.forEach((item) => {
    const node = element("div");
    node.append(element("span", "", item.label), element("strong", "", item.value));
    highlights.append(node);
  });
}

function updateEnterButton() {
  const accepted = document.querySelector("#accept-brief").checked;
  document.querySelector("#enter-room").disabled = !(accepted && appState.scenario);
}

async function enterRoom() {
  if (!appState.role || !appState.scenario) return;
  const button = document.querySelector("#enter-room");
  setBusy(button, true, "Opening room");
  try {
    appState.session = await requestJSON("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        role: appState.role,
        scenario_id: appState.scenario.scenario_id,
      }),
    });
    appState.latestRun = appState.session.runs.at(-1);
    renderWorkspace(appState.session);
    showStage(3);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(button, false, "Enter co-design room →");
  }
}

function renderWorkspace(snapshot) {
  renderParticipants(snapshot);
  renderMessages(snapshot.messages, true);
  renderParameters(snapshot.parameters);
  renderRun(snapshot.runs.at(-1), snapshot.runs.length);
  renderLedger(snapshot.decision_ledger);
  document.querySelector("#backend-pill").textContent = humanizeBackend(snapshot.llm_backend);
}

function renderParticipants(snapshot) {
  const row = document.querySelector("#participant-row");
  row.replaceChildren();
  row.append(
    element("span", "participant-chip user", `You · ${roleLabels[snapshot.user_role]}`),
    element("span", "participant-chip mediator", "B2G-Agent · mediator"),
    element("span", "participant-chip", `AI · ${roleLabels[snapshot.counterpart_role]}`),
  );
}

function renderMessages(messages, replace = false) {
  const container = document.querySelector("#messages");
  if (replace) container.replaceChildren();
  messages.forEach((message) => {
    if (container.querySelector(`[data-message-id="${message.message_id}"]`)) return;
    const wrapper = element("article", `message ${message.speaker}`);
    wrapper.dataset.messageId = message.message_id;
    const initials = message.speaker === "mediator" ? "B2G" : message.speaker === "user" ? "YOU" : "AI";
    const speakerName = message.speaker === "mediator"
      ? "B2G-Agent · mediator"
      : message.speaker === "user"
        ? `You · ${roleLabels[appState.session.user_role]}`
        : `AI · ${roleLabels[appState.session.counterpart_role]}`;
    const avatar = element("div", "message-avatar", initials);
    const body = element("div", "message-body");
    body.append(element("div", "message-meta", speakerName), element("div", "message-bubble", message.text));
    wrapper.append(avatar, body);
    container.append(wrapper);
  });
  container.scrollTop = container.scrollHeight;
}

async function sendMessage(event) {
  event.preventDefault();
  if (!appState.session) return;
  const input = document.querySelector("#message-input");
  const text = input.value.trim();
  if (!text) return;
  const optimistic = {
    message_id: `local-${Date.now()}`,
    speaker: "user",
    role: appState.role,
    text,
  };
  renderMessages([optimistic]);
  input.value = "";
  const sendButton = document.querySelector(".send-button");
  setBusy(sendButton, true, "Thinking");
  try {
    const turn = await requestJSON(`/api/sessions/${appState.session.session_id}/messages`, {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    document.querySelector(`[data-message-id="${optimistic.message_id}"]`)?.remove();
    renderMessages(turn.messages);
    appState.session.parameters = turn.parameters;
    appState.session.decision_ledger = turn.decision_ledger;
    appState.session.llm_backend = turn.llm_backend;
    if (turn.simulation_run) {
      appState.latestRun = turn.simulation_run;
      appState.session.runs.push(turn.simulation_run);
      renderRun(turn.simulation_run, appState.session.runs.length);
    }
    renderParameters(turn.parameters);
    renderLedger(turn.decision_ledger);
    document.querySelector("#backend-pill").textContent = humanizeBackend(turn.llm_backend);
  } catch (error) {
    document.querySelector(`[data-message-id="${optimistic.message_id}"]`)?.remove();
    input.value = text;
    showToast(error.message, true);
  } finally {
    setBusy(sendButton, false, "Send ↑");
    input.focus();
  }
}

async function runCurrentCase() {
  if (!appState.session) return;
  const button = document.querySelector("#rerun-button");
  setBusy(button, true, "Running");
  try {
    const run = await requestJSON(`/api/sessions/${appState.session.session_id}/simulate`, { method: "POST" });
    appState.latestRun = run;
    const snapshot = await requestJSON(`/api/sessions/${appState.session.session_id}`);
    appState.session = snapshot;
    renderRun(run, snapshot.runs.length);
    renderLedger(snapshot.decision_ledger);
    showToast(`Coupled run ${run.run_id} completed.`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(button, false, "Run current case");
  }
}

function renderParameters(parameters) {
  const list = document.querySelector("#parameter-list");
  list.replaceChildren();
  const keys = appState.scenario?.parameter_keys || Object.keys(parameters);
  keys.forEach((key) => {
    const value = parameters[key];
    const [label, unit] = parameterLabels[key] || [key, ""];
    const item = element("div");
    item.append(element("dt", "", label), element("dd", "", `${formatValue(value)} ${unit}`.trim()));
    list.append(item);
  });
}

function renderRun(run, runCount = 1) {
  if (!run) return;
  appState.latestRun = run;
  const metrics = run.metrics;
  const cards = appState.scenario.scenario_id === scenarioIds.demandResponse
    ? [
        ["Baseline peak", metrics.baseline_peak_kw_per_building, "kW/building", 2],
        ["DR target", run.parameters.dr_target_kw_per_building, "kW/building", 2],
        ["Delivered", metrics.delivered_reduction_kw_per_building, "kW/building", 2],
        ["Delivery", metrics.dr_delivery_pct, "% of target", 1],
      ]
    : [
        ["Min voltage", metrics.minimum_voltage_pu, "p.u.", 3],
        ["Line loading", metrics.maximum_line_loading_pct, "max %", 1],
        ["Transformer", metrics.maximum_transformer_loading_pct, "max %", 1],
        ["Feeder peak", metrics.feeder_peak_kw, "kW", 0],
      ];
  const grid = document.querySelector("#metric-grid");
  grid.replaceChildren();
  cards.forEach(([label, value, unit, digits]) => {
    const card = element("div", "metric-card");
    card.append(
      element("span", "", label),
      element("strong", "", Number(value).toFixed(digits)),
      element("small", "", unit),
    );
    grid.append(card);
  });
  document.querySelector("#run-count").textContent = `${runCount} ${runCount === 1 ? "run" : "runs"}`;
  const badge = document.querySelector("#feasibility-badge");
  badge.className = `feasibility-badge ${metrics.feasible ? "feasible" : "risk"}`;
  badge.textContent = metrics.feasible ? "Feasible" : "Needs revision";
  const isDemandResponse = appState.scenario.scenario_id === scenarioIds.demandResponse;
  document.querySelector("#chart-caption").textContent = isDemandResponse
    ? "24-hour counterfactual baseline versus modeled event-day load"
    : "24-hour feeder demand and 0.95 p.u. voltage constraint";
  drawLoadChart(run.timeseries, isDemandResponse);
}

function drawLoadChart(points, isDemandResponse = false) {
  const canvas = document.querySelector("#load-chart");
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(300, bounds.width);
  const height = Math.max(120, bounds.height);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 8, right: 8, top: 10, bottom: 16 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const actualValues = points.map((point) => isDemandResponse ? point.building_load_kw : point.feeder_kw);
  const baselineValues = isDemandResponse
    ? points.map((point) => point.baseline_building_load_kw)
    : actualValues;
  const maxLoad = Math.max(...actualValues, ...baselineValues) * 1.08;

  ctx.strokeStyle = "#e4ebf2";
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((fraction) => {
    const y = pad.top + chartHeight * fraction;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  });

  const coordinates = valuesToCoordinates(actualValues, maxLoad, pad, chartWidth, chartHeight);
  const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
  gradient.addColorStop(0, "rgba(23, 105, 224, 0.28)");
  gradient.addColorStop(1, "rgba(23, 105, 224, 0.01)");
  ctx.beginPath();
  coordinates.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.lineTo(coordinates.at(-1).x, height - pad.bottom);
  ctx.lineTo(coordinates[0].x, height - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();
  drawSeries(ctx, coordinates, "#1769e0", false);

  if (isDemandResponse) {
    const baselineCoordinates = valuesToCoordinates(baselineValues, maxLoad, pad, chartWidth, chartHeight);
    drawSeries(ctx, baselineCoordinates, "#0f9b76", true);
  }

  ctx.fillStyle = "#8b9aaa";
  ctx.font = "8px system-ui";
  ctx.fillText("00", pad.left, height - 3);
  ctx.fillText("12", pad.left + chartWidth / 2 - 4, height - 3);
  ctx.fillText("23", width - pad.right - 10, height - 3);
}

function valuesToCoordinates(values, maxLoad, pad, chartWidth, chartHeight) {
  return values.map((value, index) => ({
    x: pad.left + (index / (values.length - 1)) * chartWidth,
    y: pad.top + chartHeight - (value / maxLoad) * chartHeight,
  }));
}

function drawSeries(ctx, coordinates, color, dashed) {
  ctx.save();
  ctx.beginPath();
  coordinates.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.4;
  ctx.setLineDash(dashed ? [5, 4] : []);
  ctx.stroke();
  ctx.restore();
}

function renderLedger(items) {
  const list = document.querySelector("#decision-ledger");
  replaceListElement(list, items.length ? items.slice(-6) : ["No negotiated decisions yet."]);
}

async function finalizePlan() {
  if (!appState.session) return;
  const button = document.querySelector("#review-button");
  setBusy(button, true, "Preparing review");
  try {
    const plan = await requestJSON(`/api/sessions/${appState.session.session_id}/finalize`, { method: "POST" });
    appState.finalPlan = plan;
    renderFinalPlan(plan);
    showStage(4);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(button, false, "Review plan →");
  }
}

function renderFinalPlan(plan) {
  const run = plan.selected_run;
  const metrics = run.metrics;
  const isDemandResponse = run.scenario_id === scenarioIds.demandResponse;
  const status = document.querySelector("#review-status");
  status.textContent = plan.status === "ready" ? "Ready for professional review" : "Revision required";
  status.className = `review-status ${plan.status === "ready" ? "" : "risk"}`;
  document.querySelector("#final-summary").textContent = plan.summary;
  document.querySelector("#building-score").textContent = `${metrics.building_satisfaction_score.toFixed(1)}/100`;
  document.querySelector("#grid-score").textContent = `${metrics.grid_reliability_score.toFixed(1)}/100`;
  document.querySelector("#third-score-label").textContent = isDemandResponse ? "DR delivery" : "Indicative capital cost";
  document.querySelector("#capital-cost").textContent = isDemandResponse
    ? `${metrics.dr_delivery_pct.toFixed(1)}%`
    : `$${metrics.estimated_capital_cost_kusd.toFixed(0)}k`;
  const scoreBars = document.querySelectorAll(".score-row i");
  scoreBars[0].style.setProperty("--score-width", `${metrics.building_satisfaction_score}%`);
  scoreBars[1].style.setProperty("--score-width", `${metrics.grid_reliability_score}%`);
  replaceList("#accepted-decisions", plan.accepted_decisions);
  replaceList("#unresolved-items", plan.unresolved_items.length
    ? plan.unresolved_items
    : ["No unresolved hard constraints in the selected candidate."]);
  document.querySelector("#report-markdown").textContent = plan.report_markdown;
  document.querySelector("#final-disclaimer").textContent = isDemandResponse
    ? "This prototype baseline is not settlement-grade. Enrollment requires validated interval data, controls, and program rules."
    : "This deterministic backend is not a calibrated EnergyPlus/OpenDSS study. Professional approval requires validated models.";
  renderFinalConstraints(run);
}

function renderFinalConstraints(run) {
  const metrics = run.metrics;
  const constraints = run.scenario_id === scenarioIds.demandResponse
    ? [
        ["Baseline confidence ≥ 80/100", metrics.baseline_confidence_score.toFixed(1), metrics.baseline_confidence_score >= 80],
        ["Delivered reduction ≥ 90%", `${metrics.dr_delivery_pct.toFixed(1)}%`, metrics.dr_delivery_pct >= 90],
        [`Post-event rebound ≤ ${run.parameters.max_rebound_pct.toFixed(1)}%`, `${metrics.rebound_pct.toFixed(1)}%`, metrics.rebound_pct <= run.parameters.max_rebound_pct],
        ["Minimum voltage ≥ 0.95 p.u.", metrics.minimum_voltage_pu.toFixed(3), metrics.minimum_voltage_pu >= 0.95],
        ["Line and transformer loading ≤ 100%", `${Math.max(metrics.maximum_line_loading_pct, metrics.maximum_transformer_loading_pct).toFixed(1)}%`, metrics.maximum_line_loading_pct <= 100 && metrics.maximum_transformer_loading_pct <= 100],
      ]
    : [
        ["Minimum voltage ≥ 0.95 p.u.", metrics.minimum_voltage_pu.toFixed(3), metrics.minimum_voltage_pu >= 0.95],
        ["Maximum line loading ≤ 100%", `${metrics.maximum_line_loading_pct.toFixed(1)}%`, metrics.maximum_line_loading_pct <= 100],
        ["Maximum transformer loading ≤ 100%", `${metrics.maximum_transformer_loading_pct.toFixed(1)}%`, metrics.maximum_transformer_loading_pct <= 100],
      ];
  const container = document.querySelector("#final-constraints");
  container.replaceChildren();
  constraints.forEach(([label, value, pass]) => {
    const row = element("div", "constraint-row");
    row.append(element("span", "", label), element("strong", "", value), element("b", pass ? "" : "fail", pass ? "Pass" : "Fail"));
    container.append(row);
  });
}

function downloadReport() {
  if (!appState.finalPlan) return;
  const blob = new Blob([appState.finalPlan.report_markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `b2g-agent-${appState.session.session_id}-final-plan.md`;
  link.click();
  URL.revokeObjectURL(url);
}

function resetToStart() {
  appState.role = null;
  appState.scenario = null;
  appState.session = null;
  appState.latestRun = null;
  appState.finalPlan = null;

  document.querySelectorAll(".scenario-card").forEach((card) => card.classList.remove("selected"));
  document.querySelector("#scenario-detail").classList.add("is-hidden");
  document.querySelector("#accept-brief").checked = false;
  document.querySelector("#message-input").value = "";
  document.querySelector("#messages").replaceChildren();
  document.querySelector("#participant-row").replaceChildren();
  document.querySelector("#metric-grid").replaceChildren();
  document.querySelector("#parameter-list").replaceChildren();
  document.querySelector("#decision-ledger").replaceChildren(
    element("li", "", "No negotiated decisions yet."),
  );
  document.querySelector("#report-markdown").textContent = "";
  document.querySelector(".report-details")?.removeAttribute("open");
  document.querySelector("#backend-pill").textContent = "Preparing mediator…";
  updateEnterButton();
  showStage(1);
  showToast("Ready for a new role and scenario.");
}

function showStage(number) {
  document.querySelectorAll(".stage").forEach((stage) => stage.classList.remove("active"));
  document.querySelector(`#stage-${number}`).classList.add("active");
  document.querySelectorAll(".progress-step").forEach((step) => {
    const value = Number(step.dataset.step);
    step.classList.toggle("active", value === number);
    step.classList.toggle("complete", value < number);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    // A non-JSON provider or proxy error is handled below.
  }
  if (!response.ok) {
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

function replaceList(selector, items) {
  replaceListElement(document.querySelector(selector), items);
}

function replaceListElement(list, items) {
  list.replaceChildren();
  items.forEach((item) => list.append(element("li", "", item)));
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function formatValue(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(1);
  return String(value).replaceAll("_", " ");
}

function humanizeBackend(value) {
  if (!value) return "Mediator ready";
  if (value.startsWith("openai:")) return `LLM mediator · ${value.split(":")[1]}`;
  return "API not configured";
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = label;
  button.classList.toggle("loading", busy);
}

let toastTimer;
function showToast(message, error = false) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.className = `toast show ${error ? "error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.className = "toast"; }, 4200);
}
