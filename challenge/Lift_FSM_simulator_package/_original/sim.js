const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

const fsmStates = [
  "START", "DETECTION", "ALIGN", "FINE_ALIGN_CHECK", "ROTATE_PSI",
  "MOVE_LATERAL", "ROTATE_90", "REDETECT", "MOVE_FORWARD_PRE",
  "RAISE_FORK", "SLOW_FORWARD", "FIXED_FORWARD", "BOTH_LASER_CHECK",
  "ONE_LASER_CHECK", "YAW_CORRECTION", "LOWER_FORK", "PLACE_PALLET",
  "REVERSE", "DONE"
];

const sim = {
  running: false,
  editTarget: "forklift",
  dragging: false,
  world: { width: 9, height: 6.4 },
  truck: { x: 0, y: 2.0, yaw: 0, width: 4.4, depth: 1.15 },
  detection: {
    active: false,
    sampleIndex: 0,
    yawError: 0,
    xOffsetError: 0,
    forwardError: 0,
    truck: null
  },
  fork: {
    x: -1.6,
    y: -1.75,
    yaw: 18,
    width: 0.92,
    bodyLen: 1.38,
    forkLen: 1.05,
    forkGap: 0.48,
    forkWidth: 0.08,
    centerToP: 0.72,
    laserForward: 0.16,
    laserOut: 0.09
  },
  fsm: {
    state: "START",
    command: "STOP",
    timer: 0,
    stateTime: 0,
    oneLaserTime: 0,
    lateralStart: null,
    laserChangedL: false,
    laserChangedR: false,
    log: []
  },
  params: {
    moveSpeed: 0.45,
    slowSpeed: 0.16,
    rotSpeed: 55,
    standOff: 1.5,
    latTol: 0.08,
    psiTol: 2.0,
    forwardTol: 0.10,
    laserDrop: 0.03,
    laserSim: 0.02,
    fixedForwardSec: 1.0,
    raiseForkSec: 2.0,
    lowerForkSec: 1.5,
    placeSec: 1.0,
    reverseSec: 2.0,
    yawCorrectionSec: 0.45,
    stopSec: 0.25,
    autoLaser: true,
    sideShift: false,
    detNoise: true,
    detYawStd: 3.0,
    detXStd: 0.12,
    detForwardStd: 0.10,
    groundLaserHeight: 1.0,
    deckLaserHeight: 0.32
  },
  lasers: { l: 1, r: 1, baseL: 1, baseR: 1 },
  pose: null,
  canvas: null,
  ctx: null,
  lastTime: performance.now()
};

let normalSpare = null;

const el = {};

function $(id) {
  return document.getElementById(id);
}

function bindElements() {
  [
    "statePill", "cmdPill", "simCanvas", "cursorReadout", "runBtn", "stepBtn",
    "resetBtn", "dForward", "dLateral", "psiTruck", "detReadout", "laserReadout",
    "truckX", "truckY", "truckYaw", "forkX", "forkY", "forkYaw",
    "sideShiftToggle", "laserAutoToggle", "moveSpeed", "slowSpeed", "rotSpeed",
    "standOff", "reverseSec", "detNoiseToggle", "detYawStd", "detXStd", "detForwardStd",
    "resampleDetectBtn", "latTol", "psiTol", "laserDrop", "laserSim",
    "fsmList", "logBox"
  ].forEach((id) => { el[id] = $(id); });
  sim.canvas = el.simCanvas;
  sim.ctx = sim.canvas.getContext("2d");
}

function syncInputsFromState() {
  el.truckX.value = sim.truck.x.toFixed(2);
  el.truckY.value = sim.truck.y.toFixed(2);
  el.truckYaw.value = sim.truck.yaw.toFixed(1);
  el.forkX.value = sim.fork.x.toFixed(2);
  el.forkY.value = sim.fork.y.toFixed(2);
  el.forkYaw.value = sim.fork.yaw.toFixed(1);
}

function syncParamsFromInputs() {
  sim.params.moveSpeed = readNumber(el.moveSpeed, sim.params.moveSpeed);
  sim.params.slowSpeed = readNumber(el.slowSpeed, sim.params.slowSpeed);
  sim.params.rotSpeed = readNumber(el.rotSpeed, sim.params.rotSpeed);
  sim.params.standOff = readNumber(el.standOff, sim.params.standOff);
  sim.params.reverseSec = readNumber(el.reverseSec, sim.params.reverseSec);
  sim.params.latTol = readNumber(el.latTol, sim.params.latTol);
  sim.params.psiTol = readNumber(el.psiTol, sim.params.psiTol);
  sim.params.laserDrop = readNumber(el.laserDrop, sim.params.laserDrop);
  sim.params.laserSim = readNumber(el.laserSim, sim.params.laserSim);
  sim.params.sideShift = el.sideShiftToggle.checked;
  sim.params.autoLaser = el.laserAutoToggle.checked;
  sim.params.detNoise = el.detNoiseToggle.checked;
  sim.params.detYawStd = readNumber(el.detYawStd, sim.params.detYawStd);
  sim.params.detXStd = readNumber(el.detXStd, sim.params.detXStd);
  sim.params.detForwardStd = readNumber(el.detForwardStd, sim.params.detForwardStd);
}

function readNumber(input, fallback) {
  const v = Number(input.value);
  return Number.isFinite(v) ? v : fallback;
}

function wireEvents() {
  document.querySelectorAll("[data-bind]").forEach((input) => {
    input.addEventListener("input", () => {
      const [obj, key] = input.dataset.bind.split(".");
      const value = readNumber(input, sim[obj][key]);
      sim[obj][key] = key === "yaw" ? wrapDeg(value) : value;
      if (obj === "truck") updateDetectionFromCurrentErrors();
      computePose();
      draw();
    });
  });

  [
    el.moveSpeed, el.slowSpeed, el.rotSpeed, el.standOff, el.reverseSec,
    el.latTol, el.psiTol, el.laserDrop, el.laserSim,
    el.sideShiftToggle, el.laserAutoToggle,
    el.detNoiseToggle, el.detYawStd, el.detXStd, el.detForwardStd
  ].forEach((input) => {
    input.addEventListener("input", syncParamsFromInputs);
    input.addEventListener("change", syncParamsFromInputs);
  });

  el.resampleDetectBtn.addEventListener("click", () => {
    syncParamsFromInputs();
    sampleDetection("manual");
    computePose();
    draw();
  });

  el.runBtn.addEventListener("click", () => {
    sim.running = !sim.running;
    el.runBtn.textContent = sim.running ? "Pause" : "Run";
    el.runBtn.classList.toggle("running", sim.running);
  });

  el.stepBtn.addEventListener("click", () => {
    syncParamsFromInputs();
    update(1 / 12);
    draw();
  });

  el.resetBtn.addEventListener("click", resetScenario);

  document.querySelectorAll("[data-edit-target]").forEach((button) => {
    button.addEventListener("click", () => {
      sim.editTarget = button.dataset.editTarget;
      document.querySelectorAll("[data-edit-target]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
    });
  });

  sim.canvas.addEventListener("pointerdown", (event) => {
    sim.dragging = true;
    sim.canvas.setPointerCapture(event.pointerId);
    setSelectedPoseFromPointer(event);
  });
  sim.canvas.addEventListener("pointermove", (event) => {
    const p = canvasToWorld(event.offsetX, event.offsetY);
    el.cursorReadout.textContent = `x ${p.x.toFixed(2)} m, y ${p.y.toFixed(2)} m`;
    if (sim.dragging) setSelectedPoseFromPointer(event);
  });
  sim.canvas.addEventListener("pointerup", () => { sim.dragging = false; });
  sim.canvas.addEventListener("pointercancel", () => { sim.dragging = false; });
}

function setSelectedPoseFromPointer(event) {
  const p = canvasToWorld(event.offsetX, event.offsetY);
  const target = sim.editTarget === "truck" ? sim.truck : sim.fork;
  target.x = clamp(p.x, -sim.world.width / 2, sim.world.width / 2);
  target.y = clamp(p.y, -sim.world.height / 2, sim.world.height / 2);
  if (sim.editTarget === "truck") updateDetectionFromCurrentErrors();
  syncInputsFromState();
  computePose();
  draw();
}

function resetScenario() {
  sim.running = false;
  el.runBtn.textContent = "Run";
  el.runBtn.classList.remove("running");
  sim.truck = { x: 0, y: 2.0, yaw: 0, width: 4.4, depth: 1.15 };
  sim.fork = {
    x: -1.6,
    y: -1.75,
    yaw: 18,
    width: 0.92,
    bodyLen: 1.38,
    forkLen: 1.05,
    forkGap: 0.48,
    forkWidth: 0.08,
    centerToP: 0.72,
    laserForward: 0.16,
    laserOut: 0.09
  };
  sim.detection = {
    active: false,
    sampleIndex: 0,
    yawError: 0,
    xOffsetError: 0,
    forwardError: 0,
    truck: null
  };
  sim.fsm = {
    state: "START",
    command: "STOP",
    timer: 0,
    stateTime: 0,
    oneLaserTime: 0,
    lateralStart: null,
    laserChangedL: false,
    laserChangedR: false,
    log: []
  };
  sim.lasers = { l: 1, r: 1, baseL: 1, baseR: 1 };
  log("reset");
  syncInputsFromState();
  computePose();
  draw();
}

function sampleDetection(source) {
  const yawError = sim.params.detNoise ? randomNormal() * sim.params.detYawStd : 0;
  const xOffsetError = sim.params.detNoise ? randomNormal() * sim.params.detXStd : 0;
  const forwardError = sim.params.detNoise ? randomNormal() * sim.params.detForwardStd : 0;

  sim.detection.active = true;
  sim.detection.sampleIndex += 1;
  sim.detection.yawError = yawError;
  sim.detection.xOffsetError = xOffsetError;
  sim.detection.forwardError = forwardError;
  updateDetectionFromCurrentErrors();
  log(`${source} detection #${sim.detection.sampleIndex}: yaw ${yawError.toFixed(2)} deg, x ${xOffsetError.toFixed(3)} m, fwd ${forwardError.toFixed(3)} m`);
}

function updateDetectionFromCurrentErrors() {
  if (!sim.detection.active) return;
  const t = truckTangent(sim.truck);
  const n = truckNormal(sim.truck);
  const c = {
    x: sim.truck.x + t.x * sim.detection.xOffsetError - n.x * sim.detection.forwardError,
    y: sim.truck.y + t.y * sim.detection.xOffsetError - n.y * sim.detection.forwardError
  };
  sim.detection.truck = {
    x: c.x,
    y: c.y,
    yaw: wrapDeg(sim.truck.yaw + sim.detection.yawError),
    width: sim.truck.width,
    depth: sim.truck.depth
  };
}

function controlTruck() {
  return sim.detection.truck || sim.truck;
}

function randomNormal() {
  if (normalSpare !== null) {
    const value = normalSpare;
    normalSpare = null;
    return value;
  }
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const mag = Math.sqrt(-2.0 * Math.log(u));
  normalSpare = mag * Math.sin(TAU * v);
  return mag * Math.cos(TAU * v);
}

function computePose() {
  const detectedTruck = controlTruck();
  const t = truckTangent(detectedTruck);
  const n = truckNormal(detectedTruck);
  const c = { x: detectedTruck.x, y: detectedTruck.y };
  const f = forkBodyCenter();
  const pPoint = forkPoint();
  const v = { x: f.x - c.x, y: f.y - c.y };
  const dForward = vectorDot(v, n);
  const dLateral = vectorDot(v, t);
  const h = headingVector(sim.fork.yaw);
  const targetTruckYaw = angleDeg({ x: -n.x, y: -n.y });

  const lateralDir = dLateral >= 0
    ? { x: -t.x, y: -t.y }
    : { x: t.x, y: t.y };
  const targetLateralYaw = angleDeg(lateralDir);
  const psiTruck = angleDelta(sim.fork.yaw, targetTruckYaw);
  const psiLateral = angleDelta(sim.fork.yaw, targetLateralYaw);
  const psiBodyAxis = angleDelta(angleDeg({ x: -h.y, y: h.x }), angleDeg(t));
  const foot = { x: c.x + n.x * dForward, y: c.y + n.y * dForward };

  sim.pose = {
    t, n, c, f, pPoint, foot, dForward, dLateral, psiTruck, psiLateral,
    psiBodyAxis, targetTruckYaw, targetLateralYaw
  };
  updateLaserValues();
  return sim.pose;
}

function updateLaserValues() {
  if (!sim.params.autoLaser) return;
  const sensors = laserSensorPositions();
  sim.lasers.l = laserHeightAt(sensors.left);
  sim.lasers.r = laserHeightAt(sensors.right);
}

function laserSensorPositions() {
  const f = sim.fork;
  const h = headingVector(f.yaw);
  const side = { x: -h.y, y: h.x };
  const lateral = f.forkGap / 2 + f.laserOut;
  const p = forkPoint();
  const base = {
    x: p.x + h.x * f.laserForward,
    y: p.y + h.y * f.laserForward
  };
  return {
    left: {
      x: base.x + side.x * lateral,
      y: base.y + side.y * lateral
    },
    right: {
      x: base.x - side.x * lateral,
      y: base.y - side.y * lateral
    }
  };
}

function laserHeightAt(pos) {
  return isOverTruckDeck(pos) ? sim.params.deckLaserHeight : sim.params.groundLaserHeight;
}

function isOverTruckDeck(pos) {
  const t = truckTangent(sim.truck);
  const n = truckNormal(sim.truck);
  const c = { x: sim.truck.x, y: sim.truck.y };
  const v = { x: pos.x - c.x, y: pos.y - c.y };
  const forward = vectorDot(v, n);
  const lateral = vectorDot(v, t);
  const deckDepth = -forward;
  return (
    deckDepth >= 0
    && deckDepth <= sim.truck.depth
    && Math.abs(lateral) <= sim.truck.width / 2
  );
}

function update(dt) {
  syncParamsFromInputs();
  computePose();
  fsmTick(dt);
  constrainWorld();
  computePose();
  renderUi();
}

function fsmTick(dt) {
  const f = sim.fsm;
  f.stateTime += dt;

  switch (f.state) {
    case "START":
      enterState("DETECTION");
      break;
    case "DETECTION":
      command("STOP");
      enterState("ALIGN");
      break;
    case "ALIGN":
      enterState("FINE_ALIGN_CHECK");
      break;
    case "FINE_ALIGN_CHECK":
      if (Math.abs(sim.pose.dLateral) > sim.params.latTol || Math.abs(sim.pose.psiTruck) > sim.params.psiTol) {
        enterState("ROTATE_PSI");
      } else {
        enterState("MOVE_FORWARD_PRE");
      }
      break;
    case "ROTATE_PSI":
      command("ROTATE");
      rotateToward(sim.params.sideShift ? sim.pose.targetTruckYaw : sim.pose.targetLateralYaw, sim.params.rotSpeed * dt);
      if (rotatePsiSatisfied()) {
        enterStop("MOVE_LATERAL");
      }
      break;
    case "MOVE_LATERAL":
      command(sim.params.sideShift ? "SIDE" : "FWD");
      if (f.lateralStart === null) f.lateralStart = Math.abs(sim.pose.dLateral);
      moveLateralCorrection(dt);
      if (Math.abs(sim.pose.dLateral) <= sim.params.latTol || f.stateTime > lateralTimeout()) {
        enterStop("ROTATE_90");
      }
      break;
    case "ROTATE_90":
      command("ROTATE");
      rotateToward(sim.pose.targetTruckYaw, sim.params.rotSpeed * dt);
      if (Math.abs(angleDelta(sim.fork.yaw, sim.pose.targetTruckYaw)) <= sim.params.psiTol) {
        enterStop("REDETECT");
      }
      break;
    case "REDETECT":
      command("STOP");
      if (f.stateTime > 0.25) enterState("FINE_ALIGN_CHECK");
      break;
    case "MOVE_FORWARD_PRE":
      command("FWD");
      rotateToward(sim.pose.targetTruckYaw, sim.params.rotSpeed * dt * 0.35);
      moveAlongHeading(sim.params.moveSpeed * dt);
      if (sim.pose.dForward <= preInsertionCenterTarget() + sim.params.forwardTol) {
        enterStop("RAISE_FORK");
      }
      break;
    case "RAISE_FORK":
      command("RAISE_FORK");
      timedState(dt, sim.params.raiseForkSec, "SLOW_FORWARD");
      break;
    case "SLOW_FORWARD":
      command("SLOW_FWD");
      rotateToward(sim.pose.targetTruckYaw, sim.params.rotSpeed * dt * 0.2);
      moveAlongHeading(sim.params.slowSpeed * dt);
      updateLaserChangeFlags(dt);
      if (f.laserChangedL || f.laserChangedR) enterState("FIXED_FORWARD");
      break;
    case "FIXED_FORWARD":
      command("SLOW_FWD");
      moveAlongHeading(sim.params.slowSpeed * dt);
      updateLaserChangeFlags(dt);
      timedState(dt, sim.params.fixedForwardSec, "BOTH_LASER_CHECK");
      break;
    case "BOTH_LASER_CHECK":
      command("STOP");
      if (f.laserChangedL && f.laserChangedR && Math.abs(sim.lasers.l - sim.lasers.r) <= sim.params.laserSim) {
        enterState("LOWER_FORK");
      } else {
        enterState("ONE_LASER_CHECK");
      }
      break;
    case "ONE_LASER_CHECK":
      command("SLOW_FWD");
      updateLaserChangeFlags(dt);
      if (f.laserChangedL !== f.laserChangedR) {
        f.oneLaserTime += dt;
        if (f.oneLaserTime > 0.6) enterState("YAW_CORRECTION");
      } else {
        enterState("SLOW_FORWARD");
      }
      break;
    case "YAW_CORRECTION":
      command("YAW_CORRECT");
      sim.fork.yaw = wrapDeg(sim.fork.yaw + (f.laserChangedL ? -1 : 1) * sim.params.rotSpeed * 0.45 * dt);
      timedState(dt, sim.params.yawCorrectionSec, "SLOW_FORWARD");
      break;
    case "LOWER_FORK":
      command("LOWER_FORK");
      timedState(dt, sim.params.lowerForkSec, "PLACE_PALLET");
      break;
    case "PLACE_PALLET":
      command("PLACE_PALLET");
      timedState(dt, sim.params.placeSec, "REVERSE");
      break;
    case "REVERSE":
      command("BACK");
      moveAlongHeading(-sim.params.moveSpeed * dt);
      timedState(dt, sim.params.reverseSec, "DONE");
      break;
    case "DONE":
      command("STOP");
      break;
    case "STOP_INTERLOCK":
      command("STOP");
      timedState(dt, sim.params.stopSec, f.nextState || "DETECTION");
      break;
    default:
      enterState("START");
  }
}

function enterState(state) {
  if (sim.fsm.state === state) return;
  const prev = sim.fsm.state;
  sim.fsm.state = state;
  sim.fsm.stateTime = 0;
  sim.fsm.timer = 0;
  sim.fsm.lateralStart = null;
  if (state === "DETECTION" || state === "REDETECT") {
    sampleDetection(state.toLowerCase());
  }
  if (state === "SLOW_FORWARD") {
    sim.fsm.laserChangedL = false;
    sim.fsm.laserChangedR = false;
    sim.fsm.oneLaserTime = 0;
    sim.lasers.baseL = sim.lasers.l;
    sim.lasers.baseR = sim.lasers.r;
  }
  if (state === "MOVE_LATERAL") {
    sim.fsm.lateralStart = Math.abs(sim.pose.dLateral);
  }
  log(`${prev} -> ${state}`);
}

function enterStop(nextState) {
  sim.fsm.nextState = nextState;
  enterState("STOP_INTERLOCK");
}

function timedState(dt, duration, nextState) {
  sim.fsm.timer += dt;
  if (sim.fsm.timer >= duration) enterState(nextState);
}

function command(name) {
  sim.fsm.command = name;
}

function rotatePsiSatisfied() {
  const err = sim.params.sideShift ? sim.pose.psiTruck : sim.pose.psiLateral;
  return Math.abs(err) <= sim.params.psiTol;
}

function moveLateralCorrection(dt) {
  const p = sim.pose;
  if (sim.params.sideShift) {
    const sign = p.dLateral >= 0 ? -1 : 1;
    sim.fork.x += p.t.x * sign * sim.params.moveSpeed * dt;
    sim.fork.y += p.t.y * sign * sim.params.moveSpeed * dt;
  } else {
    moveAlongHeading(sim.params.moveSpeed * dt);
  }
}

function lateralTimeout() {
  const start = sim.fsm.lateralStart || Math.abs(sim.pose.dLateral);
  return Math.max(0.8, start / Math.max(0.05, sim.params.moveSpeed) + 0.4);
}

function moveAlongHeading(distance) {
  const h = headingVector(sim.fork.yaw);
  sim.fork.x += h.x * distance;
  sim.fork.y += h.y * distance;
}

function rotateToward(targetYaw, maxStep) {
  const diff = angleDelta(sim.fork.yaw, targetYaw);
  const step = clamp(diff, -maxStep, maxStep);
  sim.fork.yaw = wrapDeg(sim.fork.yaw - step);
}

function forkBodyCenter() {
  return { x: sim.fork.x, y: sim.fork.y };
}

function forkPoint() {
  const f = sim.fork;
  const h = headingVector(f.yaw);
  const centerToP = f.centerToP ?? f.bodyLen * 0.52;
  return {
    x: f.x + h.x * centerToP,
    y: f.y + h.y * centerToP
  };
}

function preInsertionCenterTarget() {
  const pose = sim.pose || computePose();
  const h = headingVector(sim.fork.yaw);
  const centerToP = sim.fork.centerToP ?? sim.fork.bodyLen * 0.52;
  const pForwardProjection = vectorDot({ x: h.x * centerToP, y: h.y * centerToP }, pose.n);
  return sim.params.standOff - pForwardProjection;
}

function updateLaserChangeFlags(dt) {
  const lDrop = sim.lasers.baseL - sim.lasers.l;
  const rDrop = sim.lasers.baseR - sim.lasers.r;
  const wasL = sim.fsm.laserChangedL;
  const wasR = sim.fsm.laserChangedR;
  if (lDrop >= sim.params.laserDrop) sim.fsm.laserChangedL = true;
  if (rDrop >= sim.params.laserDrop) sim.fsm.laserChangedR = true;
  if (!wasL && sim.fsm.laserChangedL) log(`laser L changed ${sim.lasers.baseL.toFixed(3)} -> ${sim.lasers.l.toFixed(3)}`);
  if (!wasR && sim.fsm.laserChangedR) log(`laser R changed ${sim.lasers.baseR.toFixed(3)} -> ${sim.lasers.r.toFixed(3)}`);
  if (sim.fsm.laserChangedL !== sim.fsm.laserChangedR) sim.fsm.oneLaserTime += dt;
}

function constrainWorld() {
  const hw = sim.world.width / 2;
  const hh = sim.world.height / 2;
  sim.fork.x = clamp(sim.fork.x, -hw, hw);
  sim.fork.y = clamp(sim.fork.y, -hh, hh);
  sim.truck.x = clamp(sim.truck.x, -hw, hw);
  sim.truck.y = clamp(sim.truck.y, -hh, hh);
}

function log(message) {
  const stamp = new Date().toLocaleTimeString("ko-KR", { hour12: false });
  sim.fsm.log.unshift(`${stamp}  ${message}`);
  sim.fsm.log = sim.fsm.log.slice(0, 80);
}

function renderUi() {
  const p = sim.pose || computePose();
  el.statePill.textContent = sim.fsm.state;
  el.cmdPill.textContent = sim.fsm.command;
  el.dForward.textContent = `${p.dForward.toFixed(3)} m`;
  el.dLateral.textContent = `${p.dLateral.toFixed(3)} m`;
  el.psiTruck.textContent = `${p.psiTruck.toFixed(2)} deg`;
  el.detReadout.textContent = `${sim.detection.yawError.toFixed(2)} deg / ${sim.detection.xOffsetError.toFixed(3)} / ${sim.detection.forwardError.toFixed(3)} m`;
  el.laserReadout.textContent = `${sim.lasers.l.toFixed(3)} / ${sim.lasers.r.toFixed(3)}`;
  el.logBox.innerHTML = sim.fsm.log.map((line) => `<div>${escapeHtml(line)}</div>`).join("");

  document.querySelectorAll("#fsmList li").forEach((li) => {
    li.classList.toggle("active", li.dataset.state === sim.fsm.state);
  });
  syncInputsFromState();
}

function draw() {
  resizeCanvas();
  const ctx = sim.ctx;
  ctx.clearRect(0, 0, sim.canvas.width, sim.canvas.height);
  drawGrid(ctx);
  drawTruck(ctx);
  drawDetectionEstimate(ctx);
  drawStateGeometry(ctx);
  drawForklift(ctx);
  drawLegend(ctx);
  renderUi();
}

function resizeCanvas() {
  const rect = sim.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(640, Math.floor(rect.width * dpr));
  const h = Math.max(420, Math.floor(rect.height * dpr));
  if (sim.canvas.width !== w || sim.canvas.height !== h) {
    sim.canvas.width = w;
    sim.canvas.height = h;
  }
}

function worldToCanvas(p) {
  const scale = Math.min(sim.canvas.width / sim.world.width, sim.canvas.height / sim.world.height) * 0.92;
  return {
    x: sim.canvas.width / 2 + p.x * scale,
    y: sim.canvas.height / 2 - p.y * scale,
    scale
  };
}

function canvasToWorld(x, y) {
  const c = worldToCanvas({ x: 0, y: 0 });
  return {
    x: (x - sim.canvas.width / 2) / c.scale,
    y: -(y - sim.canvas.height / 2) / c.scale
  };
}

function drawGrid(ctx) {
  const c = worldToCanvas({ x: 0, y: 0 });
  ctx.save();
  ctx.strokeStyle = "#e1e5ea";
  ctx.lineWidth = 1;
  for (let x = -Math.floor(sim.world.width / 2); x <= sim.world.width / 2; x += 0.5) {
    const a = worldToCanvas({ x, y: -sim.world.height / 2 });
    const b = worldToCanvas({ x, y: sim.world.height / 2 });
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  for (let y = -Math.floor(sim.world.height / 2); y <= sim.world.height / 2; y += 0.5) {
    const a = worldToCanvas({ x: -sim.world.width / 2, y });
    const b = worldToCanvas({ x: sim.world.width / 2, y });
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#9aa4b2";
  ctx.lineWidth = 1.4;
  line(ctx, worldToCanvas({ x: -sim.world.width / 2, y: 0 }), worldToCanvas({ x: sim.world.width / 2, y: 0 }));
  line(ctx, worldToCanvas({ x: 0, y: -sim.world.height / 2 }), worldToCanvas({ x: 0, y: sim.world.height / 2 }));
  ctx.fillStyle = "#5e6875";
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px Arial`;
  ctx.fillText("x", c.x + 42, c.y - 8);
  ctx.fillText("y", c.x + 8, c.y - 42);
  ctx.restore();
}

function drawTruck(ctx) {
  const t = truckTangent(sim.truck);
  const n = truckNormal(sim.truck);
  const c = { x: sim.truck.x, y: sim.truck.y };
  const center = { x: c.x - n.x * sim.truck.depth / 2, y: c.y - n.y * sim.truck.depth / 2 };
  const yaw = Math.atan2(t.y, t.x);
  drawRotatedRect(ctx, center, sim.truck.width, sim.truck.depth, yaw, "#eceff3", "#20242a", 3);

  const left = { x: c.x - t.x * sim.truck.width / 2, y: c.y - t.y * sim.truck.width / 2 };
  const right = { x: c.x + t.x * sim.truck.width / 2, y: c.y + t.y * sim.truck.width / 2 };
  const a = worldToCanvas(left);
  const b = worldToCanvas(right);
  ctx.save();
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 4;
  line(ctx, a, b);
  ctx.fillStyle = "#111";
  drawDot(ctx, worldToCanvas(c), 6, "#111");
  label(ctx, "C", worldToCanvas({ x: c.x + t.x * 0.08, y: c.y + t.y * 0.08 }), "#111");
  ctx.restore();
}

function drawDetectionEstimate(ctx) {
  if (!sim.detection.truck) return;
  const truck = sim.detection.truck;
  const t = truckTangent(truck);
  const n = truckNormal(truck);
  const c = { x: truck.x, y: truck.y };
  const left = { x: c.x - t.x * truck.width / 2, y: c.y - t.y * truck.width / 2 };
  const right = { x: c.x + t.x * truck.width / 2, y: c.y + t.y * truck.width / 2 };
  const normalEnd = { x: c.x + n.x * 3.8, y: c.y + n.y * 3.8 };

  ctx.save();
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = "#8c5bd6";
  ctx.lineWidth = 2.5;
  line(ctx, worldToCanvas(left), worldToCanvas(right));
  ctx.lineWidth = 1.5;
  line(ctx, worldToCanvas(c), worldToCanvas(normalEnd));
  ctx.setLineDash([]);
  drawDot(ctx, worldToCanvas(c), 5, "#8c5bd6", "#ffffff");
  label(ctx, "C_hat", worldToCanvas({ x: c.x + t.x * 0.08, y: c.y + t.y * 0.08 }), "#6a3fb5");
  ctx.restore();
}

function drawStateGeometry(ctx) {
  const p = sim.pose || computePose();
  const c = p.c;
  const foot = p.foot;
  const f = p.f;
  const pPoint = p.pPoint;
  const normalEnd = { x: c.x + p.n.x * 4.0, y: c.y + p.n.y * 4.0 };

  ctx.save();
  ctx.setLineDash([9, 7]);
  ctx.strokeStyle = "#565f6b";
  ctx.lineWidth = 1.5;
  line(ctx, worldToCanvas(c), worldToCanvas(normalEnd));
  ctx.setLineDash([]);

  ctx.strokeStyle = "#e34a4a";
  ctx.lineWidth = 3;
  arrow(ctx, worldToCanvas(c), worldToCanvas(foot), "#e34a4a");
  labelMid(ctx, "d_forward", c, foot, "#e34a4a");

  ctx.strokeStyle = "#2f6fed";
  ctx.lineWidth = 3;
  arrow(ctx, worldToCanvas(foot), worldToCanvas(f), "#2f6fed");
  labelMid(ctx, "d_lateral", foot, f, "#2f6fed");

  drawDot(ctx, worldToCanvas(foot), 5, "#111");
  label(ctx, "H", worldToCanvas({ x: foot.x + 0.07, y: foot.y + 0.07 }), "#111");
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "#6d7480";
  ctx.lineWidth = 1.5;
  line(ctx, worldToCanvas(f), worldToCanvas(pPoint));
  ctx.setLineDash([]);

  drawPsiArc(ctx, p);
  ctx.restore();
}

function drawForklift(ctx) {
  const f = sim.fork;
  const yaw = f.yaw * DEG;
  const centerWorld = forkBodyCenter();
  const pWorld = forkPoint();
  const pCanvas = worldToCanvas(pWorld);
  const scale = pCanvas.scale;

  ctx.save();
  ctx.translate(pCanvas.x, pCanvas.y);
  ctx.rotate(Math.PI / 2 - yaw);

  const bodyW = f.width * scale;
  const bodyL = f.bodyLen * scale;
  const frontY = 0.05 * scale;
  const rearY = bodyL;
  const halfW = bodyW / 2;

  drawLocalRect(ctx, -f.forkGap * scale / 2 - f.forkWidth * scale / 2, -f.forkLen * scale, f.forkWidth * scale, f.forkLen * scale, "#252a31", "#0b0d10");
  drawLocalRect(ctx, f.forkGap * scale / 2 - f.forkWidth * scale / 2, -f.forkLen * scale, f.forkWidth * scale, f.forkLen * scale, "#252a31", "#0b0d10");

  ctx.fillStyle = "#e8b923";
  ctx.strokeStyle = "#171b21";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(-halfW, frontY);
  ctx.lineTo(halfW, frontY);
  ctx.lineTo(halfW, rearY - 0.18 * scale);
  ctx.quadraticCurveTo(halfW * 0.72, rearY + 0.12 * scale, 0, rearY + 0.14 * scale);
  ctx.quadraticCurveTo(-halfW * 0.72, rearY + 0.12 * scale, -halfW, rearY - 0.18 * scale);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  drawLocalRect(ctx, -halfW - 0.10 * scale, 0.18 * scale, 0.12 * scale, 0.42 * scale, "#252a31", "#111");
  drawLocalRect(ctx, halfW - 0.02 * scale, 0.18 * scale, 0.12 * scale, 0.42 * scale, "#252a31", "#111");
  drawLocalRect(ctx, -halfW - 0.08 * scale, 0.82 * scale, 0.12 * scale, 0.34 * scale, "#252a31", "#111");
  drawLocalRect(ctx, halfW - 0.04 * scale, 0.82 * scale, 0.12 * scale, 0.34 * scale, "#252a31", "#111");

  const sensorY = -f.laserForward * scale;
  const sensorX = (f.forkGap / 2 + f.laserOut) * scale;
  drawLocalSensor(ctx, -sensorX, sensorY, sim.lasers.l, "L");
  drawLocalSensor(ctx, sensorX, sensorY, sim.lasers.r, "R");

  ctx.restore();

  ctx.save();
  drawDot(ctx, worldToCanvas(pWorld), 5, "#ffffff", "#111");
  label(ctx, "P", worldToCanvas({ x: pWorld.x + 0.08, y: pWorld.y + 0.08 }), "#111");
  drawDot(ctx, worldToCanvas(centerWorld), 5, "#111", "#ffffff");
  label(ctx, "O", worldToCanvas({ x: centerWorld.x + 0.08, y: centerWorld.y + 0.08 }), "#111");
  ctx.restore();
}

function drawLocalSensor(ctx, x, y, value, labelText) {
  const changed = value <= (sim.params.groundLaserHeight - sim.params.laserDrop);
  const r = 7 * (window.devicePixelRatio || 1);
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, TAU);
  ctx.fillStyle = changed ? "#e34a4a" : "#22a06b";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#111";
  ctx.font = `${10 * (window.devicePixelRatio || 1)}px Arial`;
  ctx.fillText(labelText, x + r + 2, y - r);
  ctx.restore();
}

function drawLocalRect(ctx, x, y, w, h, fill, stroke) {
  ctx.save();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawLocalArrow(ctx, x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 10 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(angle - 0.48), y2 - head * Math.sin(angle - 0.48));
  ctx.lineTo(x2 - head * Math.cos(angle + 0.48), y2 - head * Math.sin(angle + 0.48));
  ctx.closePath();
  ctx.fill();
}

function drawPsiArc(ctx, p) {
  const center = worldToCanvas(p.f);
  const radius = 54 * (window.devicePixelRatio || 1);
  const a0 = -sim.fork.yaw * DEG;
  const target = p.targetTruckYaw;
  const a1 = -target * DEG;
  ctx.save();
  ctx.strokeStyle = "#e57924";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, a0, a1, angleDelta(sim.fork.yaw, target) > 0);
  ctx.stroke();
  label(ctx, "psi_truck", { x: center.x + radius * 0.44, y: center.y - radius * 0.62 }, "#e57924");
  ctx.restore();
}

function drawLegend(ctx) {
  const dpr = window.devicePixelRatio || 1;
  const x = sim.canvas.width - 250 * dpr;
  const y = 22 * dpr;
  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.strokeStyle = "#d6dae1";
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, 218 * dpr, 148 * dpr, 8 * dpr);
  ctx.fill();
  ctx.stroke();
  ctx.font = `${12 * dpr}px Arial`;
  const rows = [
    ["C", "loading face center", "#111"],
    ["H", "foot point", "#111"],
    ["O", "forklift center", "#111"],
    ["P", "fork inner midpoint", "#111"],
    ["purple", "detected face", "#6a3fb5"],
    ["red", "d_forward", "#e34a4a"],
    ["blue", "d_lateral", "#2f6fed"],
    ["orange", "psi_truck", "#e57924"]
  ];
  rows.forEach((row, i) => {
    ctx.fillStyle = row[2];
    ctx.fillText(`${row[0]}: ${row[1]}`, x + 14 * dpr, y + (23 + i * 15) * dpr);
  });
  ctx.restore();
}

function drawRotatedRect(ctx, center, width, height, yawRad, fill, stroke, lineWidth) {
  const c = worldToCanvas(center);
  const scale = c.scale;
  ctx.save();
  ctx.translate(c.x, c.y);
  ctx.rotate(-yawRad);
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.rect(-width * scale / 2, -height * scale / 2, width * scale, height * scale);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function line(ctx, a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function arrow(ctx, a, b, color) {
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const head = 10 * (window.devicePixelRatio || 1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  line(ctx, a, b);
  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x - head * Math.cos(angle - 0.48), b.y - head * Math.sin(angle - 0.48));
  ctx.lineTo(b.x - head * Math.cos(angle + 0.48), b.y - head * Math.sin(angle + 0.48));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawDot(ctx, p, r, fill, stroke) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x, p.y, r * (window.devicePixelRatio || 1), 0, TAU);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
}

function label(ctx, text, p, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = `${13 * (window.devicePixelRatio || 1)}px Arial`;
  ctx.fillText(text, p.x + 6, p.y - 6);
  ctx.restore();
}

function labelMid(ctx, text, a, b, color) {
  const mid = worldToCanvas({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  label(ctx, text, mid, color);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function truckTangent(truck = controlTruck()) {
  return headingVector(truck.yaw);
}

function truckNormal(truck = controlTruck()) {
  const t = truckTangent(truck);
  return { x: t.y, y: -t.x };
}

function headingVector(yawDeg) {
  const a = yawDeg * DEG;
  return { x: Math.cos(a), y: Math.sin(a) };
}

function angleDeg(v) {
  return wrapDeg(Math.atan2(v.y, v.x) * RAD);
}

function vectorDot(a, b) {
  return a.x * b.x + a.y * b.y;
}

function angleDelta(current, target) {
  return wrapDeg(current - target);
}

function wrapDeg(deg) {
  let v = ((deg + 180) % 360 + 360) % 360 - 180;
  if (v === -180) v = 180;
  return v;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[ch]));
}

function initFsmList() {
  el.fsmList.innerHTML = fsmStates.map((state) => `<li data-state="${state}">${state}</li>`).join("");
}

function loop(now) {
  const dt = Math.min(0.05, (now - sim.lastTime) / 1000 || 0);
  sim.lastTime = now;
  if (sim.running) update(dt);
  draw();
  requestAnimationFrame(loop);
}

function init() {
  bindElements();
  initFsmList();
  wireEvents();
  syncInputsFromState();
  syncParamsFromInputs();
  computePose();
  log("ready");
  requestAnimationFrame(loop);
}

window.addEventListener("resize", draw);
init();
