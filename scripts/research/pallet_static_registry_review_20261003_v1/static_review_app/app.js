"use strict";

const severityOptions = [
  ["clean", "Clean"], ["moderate", "중간 가림"],
  ["severe", "심한 가림"], ["unknown", "Unknown"],
];
const visibilityOptions = [
  ["", "미검수"], ["DIRECT_VISIBLE", "직접 보임"],
  ["EXTERNAL_OCCLUDED", "외부 물체 가림"], ["SELF_OCCLUDED", "자체 가림"],
  ["OUT_OF_FRAME", "화면 밖"], ["OBJECT_ABSENT", "객체 없음"],
  ["UNKNOWN", "판정 불가"],
];

const ui = Object.fromEntries([
  "reviewer", "saveStatus", "prev", "next", "position", "caseId", "frame",
  "showOverlay", "zoom", "exposure", "metadata", "severityField", "severityHint",
  "severityButtons", "unknownWrap", "unknownReason", "corners", "save",
  "nextPending", "severityProgress", "visibilityProgress", "export", "importFile",
  "submit", "dialog", "dialogText", "dialogClose",
].map(id => [id, document.getElementById(id)]));

let manifest;
let state;
let index = 0;

function responseFor(caseId) {
  state.responses[caseId] ??= {corners: {}};
  state.responses[caseId].corners ??= {};
  return state.responses[caseId];
}

function showMessage(text) {
  ui.dialogText.textContent = text;
  ui.dialog.showModal();
}

function selectedSeverity(response, value) {
  response.frame_severity = value;
  if (value !== "unknown") delete response.frame_unknown_reason;
  render();
}

function renderSeverity(current, response) {
  ui.severityButtons.replaceChildren();
  const locked = current.frame_severity.locked;
  ui.severityField.disabled = locked;
  ui.severityHint.textContent = locked
    ? `기존 사람 판정(잠금): ${current.frame_severity.status}`
    : "원본 RGB를 보고 한 등급을 선택하세요.";
  const value = locked ? current.frame_severity.status : response.frame_severity;
  for (const [code, label] of severityOptions) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = value === code ? "selected" : "";
    button.disabled = locked;
    button.addEventListener("click", () => selectedSeverity(response, code));
    ui.severityButtons.append(button);
  }
  ui.unknownWrap.classList.toggle("hidden", locked || value !== "unknown");
  ui.unknownReason.value = response.frame_unknown_reason || "";
}

function renderCorners(current, response) {
  ui.corners.replaceChildren();
  for (const corner of current.corners) {
    const row = document.createElement("div");
    row.className = "corner-row";
    const label = document.createElement("label");
    label.textContent = `C${corner.corner_id} · ${corner.metric_reference ? "평가점" : "참조 없음"}`;
    const select = document.createElement("select");
    select.disabled = corner.locked;
    const currentValue = corner.locked ? corner.status : (response.corners[String(corner.corner_id)] || "");
    for (const [code, text] of visibilityOptions) {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = text;
      option.selected = code === currentValue;
      select.append(option);
    }
    select.addEventListener("change", () => {
      if (select.value) response.corners[String(corner.corner_id)] = select.value;
      else delete response.corners[String(corner.corner_id)];
      updateProgress();
    });
    const badge = document.createElement("span");
    badge.className = corner.locked ? "badge locked" : "badge";
    badge.textContent = corner.locked ? "잠금" : (corner.metric_reference ? "필수 집계" : "선택");
    row.append(label, select, badge);
    ui.corners.append(row);
  }
}

function render() {
  const current = manifest.cases[index];
  const response = responseFor(current.case_id);
  ui.position.textContent = `${index + 1} / ${manifest.cases.length}`;
  ui.caseId.textContent = current.case_id;
  ui.prev.disabled = index === 0;
  ui.next.disabled = index === manifest.cases.length - 1;
  ui.frame.src = `/asset/${index}/${response.overlay_exposed ? "overlay" : "image"}`;
  ui.showOverlay.disabled = Boolean(response.overlay_exposed);
  ui.exposure.textContent = response.overlay_exposed
    ? "참조 오버레이 열람 기록됨" : "아직 참조 오버레이를 열지 않음";
  ui.metadata.innerHTML = `<div><dt>세션</dt><dd>${current.session}</dd></div>` +
    `<div><dt>재질</dt><dd>${current.material}</dd></div>` +
    `<div><dt>형상</dt><dd>${current.shape}</dd></div>`;
  renderSeverity(current, response);
  renderCorners(current, response);
  updateProgress();
}

function progressCounts() {
  let severityDone = 0;
  let visibilityDone = 0;
  let visibilityTotal = 0;
  for (const item of manifest.cases) {
    const response = state.responses[item.case_id] || {corners: {}};
    if (item.population === "GREEN0918" && severityOptions.some(([code]) => code === response.frame_severity)) severityDone++;
    for (const corner of item.corners) {
      if (corner.locked || !corner.metric_reference) continue;
      visibilityTotal++;
      if (visibilityOptions.some(([code]) => code && code === (response.corners || {})[String(corner.corner_id)])) visibilityDone++;
    }
  }
  return {severityDone, visibilityDone, visibilityTotal};
}

function updateProgress() {
  const p = progressCounts();
  ui.severityProgress.textContent = `${p.severityDone} / 119`;
  ui.visibilityProgress.textContent = `${p.visibilityDone} / ${p.visibilityTotal}`;
}

async function save() {
  state.reviewer = ui.reviewer.value;
  const current = manifest.cases[index];
  const response = responseFor(current.case_id);
  if (response.frame_severity === "unknown") response.frame_unknown_reason = ui.unknownReason.value.trim();
  const request = await fetch("/save", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(state)});
  const result = await request.json();
  if (!request.ok) throw new Error(result.reason || "저장 실패");
  ui.saveStatus.textContent = `초안 저장됨 · ${new Date().toLocaleTimeString()}`;
}

function nextPending() {
  for (let offset = 1; offset <= manifest.cases.length; offset++) {
    const candidate = (index + offset) % manifest.cases.length;
    const item = manifest.cases[candidate];
    const response = state.responses[item.case_id] || {corners: {}};
    const pendingSeverity = item.population === "GREEN0918" && !response.frame_severity;
    const pendingCorner = item.corners.some(c => c.metric_reference && !c.locked && !(response.corners || {})[String(c.corner_id)]);
    if (pendingSeverity || pendingCorner) { index = candidate; render(); return; }
  }
  showMessage("현재 범위의 미검수 항목이 없습니다.");
}

async function init() {
  [manifest, state] = await Promise.all([fetch("/manifest").then(r => r.json()), fetch("/state").then(r => r.json())]);
  ui.reviewer.value = state.reviewer || "";
  ui.saveStatus.textContent = `${state.review_status || "DRAFT_NOT_HUMAN_REVIEWED"} · ${state.manifest_sha256.slice(0, 12)}`;
  render();
}

ui.prev.addEventListener("click", () => { if (index > 0) { index--; render(); } });
ui.next.addEventListener("click", () => { if (index + 1 < manifest.cases.length) { index++; render(); } });
ui.showOverlay.addEventListener("click", () => { responseFor(manifest.cases[index].case_id).overlay_exposed = true; render(); });
ui.zoom.addEventListener("click", () => {
  ui.frame.classList.toggle("zoomed");
  ui.zoom.textContent = ui.frame.classList.contains("zoomed") ? "원본 맞춤" : "확대";
});
ui.unknownReason.addEventListener("input", () => { responseFor(manifest.cases[index].case_id).frame_unknown_reason = ui.unknownReason.value; });
ui.save.addEventListener("click", () => save().catch(e => showMessage(e.message)));
ui.nextPending.addEventListener("click", nextPending);
ui.export.addEventListener("click", () => { window.location.href = "/export"; });
ui.importFile.addEventListener("change", async () => {
  try {
    const candidate = JSON.parse(await ui.importFile.files[0].text());
    const request = await fetch("/import", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(candidate)});
    const result = await request.json();
    if (!request.ok) throw new Error(result.reason || "가져오기 실패");
    state = await fetch("/state").then(r => r.json());
    ui.reviewer.value = state.reviewer || "";
    render();
    showMessage("JSON 초안을 검증하고 불러왔습니다.");
  } catch (error) { showMessage(error.message); }
  ui.importFile.value = "";
});
ui.submit.addEventListener("click", async () => {
  try {
    await save();
    if (!window.confirm("119개 정사각형 가림 판정을 사람 검수 제출본으로 확정할까요?")) return;
    const request = await fetch("/submit", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(state)});
    const result = await request.json();
    if (!request.ok) throw new Error(result.reason || "제출 실패");
    showMessage(`제출본이 생성되었습니다.\n${result.output}`);
  } catch (error) { showMessage(error.message); }
});
ui.dialogClose.addEventListener("click", () => ui.dialog.close());

init().catch(error => showMessage(`초기화 실패: ${error.message}`));
