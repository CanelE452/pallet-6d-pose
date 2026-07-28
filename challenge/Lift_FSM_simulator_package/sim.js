const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

// PDF (그림 2 / 그림 11) + 25y_automatic_lifter align.py snapshot-based FSM.
// *_CHECK 상태 진입 시 perception (psi/d_lat/d_fwd) 1회 snapshot → plan compute.
// 운동 상태 (YAW_CORRECT_*, ALIGN_*_ADJUST, LATERAL_*, FORWARD_AFTER_*, INSERT)
// 진행 중 perception 무시. IMU(fork.yaw 변화)/timer 만으로 종료 판정.
const fsmStates = [
  "START",
  "SEARCH",
  "YAW_CHECK", "YAW_CORRECT_RIGHT", "YAW_CORRECT_LEFT",
  "OFFSET_CHECK",
  "DIST_CHECK", "ALIGN_FWD_ADJUST", "ALIGN_BWD_ADJUST",
  "LATERAL_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT", "LATERAL_ROTATE_LEFT_BACK",
  "LATERAL_ROTATE_LEFT",  "FORWARD_AFTER_LEFT",  "LATERAL_ROTATE_RIGHT_BACK",
  "STOP_INTERLOCK",
  "READY_TO_INSERT",
  "INSERT",
  "DONE"
];

const sim = {
  running: false,
  editTarget: "forklift",
  dragging: false,
  world: { width: 9, height: 6.4 },
  pallet: {
    x: 0,
    y: 2.05,
    yaw: 0,
    width: 1.10,
    depth: 1.30,
    height: 0.12,
    forkOpening: 0.32,
    forkOpeningHeight: 0.085
  },
  detection: {
    active: false,
    sampleIndex: 0,
    yawError: 0,
    xOffsetError: 0,
    forwardError: 0,
    reprojError: 0,
    inFov: true,
    accepted: true,
    pallet: null,
    source: "noise"
  },
  fork: {
    x: -0.5,
    y: -0.9,
    yaw: 80,
    width: 0.92,
    bodyLen: 1.38,
    forkLen: 1.05,
    forkGap: 0.30,
    forkWidth: 0.06,
    centerToP: 0.72,
    laserForward: 0.16,
    laserOut: 0.05
  },
  camera: {
    mountForward: 0.45,
    mountHeight: 0.85,
    // RealSense D435I 마스트 마운트 tilt (OpenCV camera convention)
    //   pitch > 0 = down-tilt (광축이 아래로 기울어짐, 일반 마운트)
    //   roll  > 0 = viewer 기준 시계방향 회전 (보통 0, 미스얼라인 검증용)
    mountPitchDeg: 10.0,
    mountRollDeg:  0.0,
    fovHorizontalDeg: 87,
    rangeMin: 0.30,
    rangeMax: 3.5,
    show: true
  },
  fsm: {
    state: "START",
    command: "STOP",
    timer: 0,
    stateTime: 0,
    lateralStart: null,
    log: [],
    // snapshot (CHECK 진입 시 1회 캡쳐) + plan (cmd 진행 중 사용)
    snapshot: null,        // { psi, dLateral, dForward, ts }
    yawPlan: null,         // { direction, yawAbsDeg, refYaw }
    fwdPlan: null,         // { direction, distanceM, fwdSec, deadlineTs }
    lateralPlan: null,     // { direction, yawAbsDeg, fwdSec, backYawDeg, refYaw, deadlineTs }
    insertPlan: null,      // { fwdSec, deadlineTs }
    nextState: null
  },
  params: {
    // 운동 모델 — 25y_automatic_lifter/depth_cam/calib/config.py 와 일치.
    // moveSpeed = vmax = FWD_A·FWD_T1 (정속). dynamics 는 가속+정속 piecewise.
    moveSpeed: 0.306,        // vmax (m/s)
    slowSpeed: 0.25,         // INSERT_FWD_MPS (정속, INSERT 전용)
    rotSpeed: 55,            // (시뮬용. IMU rel_yaw 종료라 cycle 정합에 영향 X)
    standOff: 1.0,           // → alignDistM 으로 매핑
    latTol: 0.12,            // OFF_TOL_M
    psiTol: 2.0,             // YAW_TOL_DEG
    forwardTol: 0.08,
    raiseForkSec: 1.5,
    reverseSec: 2.0,
    stopSec: 1.2,            // STOP_SEC
    reprojAccept: 5.0,
    sideShift: false,
    detNoise: true,
    detYawStd: 1.0,
    detXStd: 0.02,
    detForwardStd: 0.02,
    detReprojBase: 2.0,
    detReprojStd: 1.5,
    // ===== PDF / align.py 매핑 (snapshot-based FSM) =====
    // ALIGN_DIST_M : DIST_CHECK 목표 d_forward (entry face).
    //                config.py 는 2.20m, 시뮬 캔버스 제약상 default 1.0m.
    alignDistM: 1.0,
    alignBandM: 0.10,
    lateralBackYawDeg: 85.0, // LATERAL_BACK_YAW_DEG
    yawTurnMinDeg: 1.0,      // YAW_TURN_MIN_DEG
    insertPocketM: 0.50,           // (deprecated)
    insertSafetyBackM: 0.10,       // (deprecated, 옛 fork-tip 기준)
    insertBodySafetyM: 0.05,       // INSERT_BODY_SAFETY_M: forklift body 가 pallet entry face 에서 띄울 거리 (m)
    // ===== motion_models.py 의 t(d) piecewise 모델 (config.py 파라미터) =====
    // d ≤ d_acc : t = t0 + sqrt(2d/a)
    // d >  d_acc: t = t0 + t1 + (d - d_acc)/vmax
    // d_acc = 0.5·a·t1² ≈ 0.655m, vmax = a·t1 ≈ 0.306 m/s
    fwdT0: -0.0202,
    fwdT1: 4.278,
    fwdA: 0.07156,
    fwdMinSec: 1.0,          // FWD_MIN_SEC
    fwdMaxSec: 15.0          // FWD_MAX_SEC
  },
  pose: null,
  canvas: null,
  ctx: null,
  lastTime: performance.now(),
  // DOPE-style ψ (pose6d_adapter.py reproduction). UI toggle 로 FSM 입력 선택.
  dopeFsm: false,
  dopePose: null   // { psi, dLat, dFwd, R, t }
};

let normalSpare = null;
const el = {};

function $(id) { return document.getElementById(id); }

function bindElements() {
  [
    "statePill", "cmdPill", "simCanvas", "cursorReadout", "runBtn", "stepBtn",
    "resetBtn", "dForward", "dLateral", "psiPallet", "detReadout", "fovReadout",
    "palletX", "palletY", "palletYaw", "palletWidth", "palletDepth", "forkX", "forkY", "forkYaw",
    "sideShiftToggle", "cameraShowToggle", "mountPitchDeg", "mountRollDeg",
    "moveSpeed", "slowSpeed", "rotSpeed",
    "standOff", "alignBandM", "lateralBackYawDeg", "insertSafetyBackM", "insertBodySafetyM",
    "detNoiseToggle", "detYawStd", "detXStd", "detForwardStd",
    "reprojAccept", "resampleDetectBtn", "latTol", "psiTol",
    "jsonDropZone", "jsonFileInput", "jsonStatus",
    "fsmList", "logBox", "fsmDiagramCanvas",
    "useDopePsiToggle", "psiGeometric", "psiDope",
    "dopeDLat", "dopeDFwd"
  ].forEach((id) => { el[id] = $(id); });
  sim.canvas = el.simCanvas;
  sim.ctx = sim.canvas.getContext("2d");
  if (el.fsmDiagramCanvas) {
    sim.diagramCanvas = el.fsmDiagramCanvas;
    sim.diagramCtx = sim.diagramCanvas.getContext("2d");
  }
}

function syncInputsFromState() {
  el.palletX.value = sim.pallet.x.toFixed(2);
  el.palletY.value = sim.pallet.y.toFixed(2);
  el.palletYaw.value = sim.pallet.yaw.toFixed(1);
  if (el.palletWidth) el.palletWidth.value = sim.pallet.width.toFixed(2);
  if (el.palletDepth) el.palletDepth.value = sim.pallet.depth.toFixed(2);
  el.forkX.value = sim.fork.x.toFixed(2);
  el.forkY.value = sim.fork.y.toFixed(2);
  el.forkYaw.value = sim.fork.yaw.toFixed(1);
}

function applyPalletPreset(kind) {
  if (kind === "long") {
    sim.pallet.width = 1.30;
    sim.pallet.depth = 1.10;
  } else if (kind === "short") {
    sim.pallet.width = 1.10;
    sim.pallet.depth = 1.30;
  }
  syncInputsFromState();
  computePose();
  draw();
}

function syncParamsFromInputs() {
  sim.params.moveSpeed = readNumber(el.moveSpeed, sim.params.moveSpeed);
  sim.params.slowSpeed = readNumber(el.slowSpeed, sim.params.slowSpeed);
  sim.params.rotSpeed = readNumber(el.rotSpeed, sim.params.rotSpeed);
  // "Stand-off" 입력은 새 FSM 에서 ALIGN_DIST_M 로 reuse.
  sim.params.alignDistM = readNumber(el.standOff, sim.params.alignDistM);
  sim.params.standOff = sim.params.alignDistM;
  if (el.alignBandM) sim.params.alignBandM = readNumber(el.alignBandM, sim.params.alignBandM);
  if (el.lateralBackYawDeg) sim.params.lateralBackYawDeg = readNumber(el.lateralBackYawDeg, sim.params.lateralBackYawDeg);
  if (el.insertSafetyBackM) sim.params.insertSafetyBackM = readNumber(el.insertSafetyBackM, sim.params.insertSafetyBackM);
  if (el.insertBodySafetyM) sim.params.insertBodySafetyM = readNumber(el.insertBodySafetyM, sim.params.insertBodySafetyM);
  sim.params.latTol = readNumber(el.latTol, sim.params.latTol);
  sim.params.psiTol = readNumber(el.psiTol, sim.params.psiTol);
  sim.params.reprojAccept = readNumber(el.reprojAccept, sim.params.reprojAccept);
  sim.params.sideShift = el.sideShiftToggle.checked;
  sim.params.detNoise = el.detNoiseToggle.checked;
  sim.params.detYawStd = readNumber(el.detYawStd, sim.params.detYawStd);
  sim.params.detXStd = readNumber(el.detXStd, sim.params.detXStd);
  sim.params.detForwardStd = readNumber(el.detForwardStd, sim.params.detForwardStd);
  sim.camera.show = el.cameraShowToggle.checked;
  if (el.mountPitchDeg) sim.camera.mountPitchDeg = readNumber(el.mountPitchDeg, sim.camera.mountPitchDeg);
  if (el.mountRollDeg)  sim.camera.mountRollDeg  = readNumber(el.mountRollDeg,  sim.camera.mountRollDeg);
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
      if (obj === "pallet") updateDetectionFromCurrentErrors();
      computePose();
      draw();
    });
  });

  [
    el.moveSpeed, el.slowSpeed, el.rotSpeed, el.standOff,
    el.alignBandM, el.lateralBackYawDeg, el.insertSafetyBackM, el.insertBodySafetyM,
    el.latTol, el.psiTol, el.reprojAccept,
    el.sideShiftToggle, el.cameraShowToggle,
    el.mountPitchDeg, el.mountRollDeg,
    el.detNoiseToggle, el.detYawStd, el.detXStd, el.detForwardStd
  ].forEach((input) => {
    if (!input) return;
    input.addEventListener("input", () => { syncParamsFromInputs(); computePose(); draw(); });
    input.addEventListener("change", () => { syncParamsFromInputs(); computePose(); draw(); });
  });

  el.resampleDetectBtn.addEventListener("click", () => {
    syncParamsFromInputs();
    sampleDetection("manual");
    computePose();
    draw();
  });

  if (el.useDopePsiToggle) {
    el.useDopePsiToggle.addEventListener("change", () => {
      sim.dopeFsm = el.useDopePsiToggle.checked;
      log(`DOPE-converted psi FSM input: ${sim.dopeFsm ? "ON" : "OFF"}`);
      computePose();
      draw();
    });
  }

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

  document.querySelectorAll("[data-pallet-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      applyPalletPreset(button.dataset.palletPreset);
      document.querySelectorAll("[data-pallet-preset]").forEach((b) => b.classList.remove("active"));
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

  wireJsonDrop();
}

function wireJsonDrop() {
  const zone = el.jsonDropZone;
  if (!zone) return;
  ["dragenter", "dragover"].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
    });
  });
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) loadJsonFile(file);
  });
  zone.addEventListener("click", () => el.jsonFileInput.click());
  el.jsonFileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) loadJsonFile(file);
  });
}

function loadJsonFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      applyDopeDetection(data);
      el.jsonStatus.textContent = `loaded: ${file.name}`;
      el.jsonStatus.classList.remove("error");
    } catch (err) {
      el.jsonStatus.textContent = `parse error: ${err.message}`;
      el.jsonStatus.classList.add("error");
    }
  };
  reader.readAsText(file);
}

function applyDopeDetection(data) {
  let det = null;
  if (data && data.pallet) {
    det = data.pallet;
  } else if (data && Array.isArray(data.objects) && data.objects.length > 0) {
    const obj = data.objects[0];
    det = {
      x: Number(obj.x ?? obj.location?.[0] ?? 0),
      y: Number(obj.y ?? obj.location?.[2] ?? 0),
      yaw: Number(obj.yaw ?? 0),
      reproj_error: Number(obj.reproj_error ?? obj.projected_cuboid_error ?? 0)
    };
  }
  if (!det) {
    log("json: no pallet field found");
    return;
  }
  sim.detection.source = "json";
  sim.detection.active = true;
  sim.detection.sampleIndex += 1;
  sim.detection.yawError = wrapDeg(Number(det.yaw) - sim.pallet.yaw);
  sim.detection.xOffsetError = Number(det.x) - sim.pallet.x;
  sim.detection.forwardError = Number(det.y) - sim.pallet.y;
  sim.detection.reprojError = Number(det.reproj_error ?? 0);
  sim.detection.pallet = {
    x: Number(det.x),
    y: Number(det.y),
    yaw: wrapDeg(Number(det.yaw)),
    width: sim.pallet.width,
    depth: sim.pallet.depth,
    height: sim.pallet.height
  };
  evaluateDetectionAcceptance();
  log(`json detection: yaw ${det.yaw} deg, x ${det.x} m, y ${det.y} m, reproj ${sim.detection.reprojError.toFixed(2)} px`);
  computePose();
  draw();
}

function setSelectedPoseFromPointer(event) {
  const p = canvasToWorld(event.offsetX, event.offsetY);
  const target = sim.editTarget === "pallet" ? sim.pallet : sim.fork;
  target.x = clamp(p.x, -sim.world.width / 2, sim.world.width / 2);
  target.y = clamp(p.y, -sim.world.height / 2, sim.world.height / 2);
  if (sim.editTarget === "pallet") updateDetectionFromCurrentErrors();
  syncInputsFromState();
  computePose();
  draw();
}

function resetScenario() {
  sim.running = false;
  el.runBtn.textContent = "Run";
  el.runBtn.classList.remove("running");
  sim.pallet = { x: 0, y: 2.05, yaw: 0, width: 1.10, depth: 1.30, height: 0.12, forkOpening: 0.32, forkOpeningHeight: 0.085 };
  sim.fork = {
    x: -0.5,
    y: -0.9,
    yaw: 80,
    width: 0.92,
    bodyLen: 1.38,
    forkLen: 1.05,
    forkGap: 0.30,
    forkWidth: 0.06,
    centerToP: 0.72,
    laserForward: 0.16,
    laserOut: 0.05
  };
  sim.detection = {
    active: false,
    sampleIndex: 0,
    yawError: 0,
    xOffsetError: 0,
    forwardError: 0,
    reprojError: 0,
    inFov: true,
    accepted: true,
    pallet: null,
    source: "noise"
  };
  sim.fsm = {
    state: "START",
    command: "STOP",
    timer: 0,
    stateTime: 0,
    lateralStart: null,
    log: [],
    snapshot: null,
    yawPlan: null,
    fwdPlan: null,
    lateralPlan: null,
    insertPlan: null,
    nextState: null
  };
  if (el.jsonStatus) {
    el.jsonStatus.textContent = "no file loaded";
    el.jsonStatus.classList.remove("error");
  }
  log("reset");
  syncInputsFromState();
  computePose();
  draw();
}

function sampleDetection(source) {
  const yawError = sim.params.detNoise ? randomNormal() * sim.params.detYawStd : 0;
  const xOffsetError = sim.params.detNoise ? randomNormal() * sim.params.detXStd : 0;
  const forwardError = sim.params.detNoise ? randomNormal() * sim.params.detForwardStd : 0;
  const reprojError = Math.max(0, sim.params.detReprojBase + (sim.params.detNoise ? Math.abs(randomNormal()) * sim.params.detReprojStd : 0));

  sim.detection.source = "noise";
  sim.detection.active = true;
  sim.detection.sampleIndex += 1;
  sim.detection.yawError = yawError;
  sim.detection.xOffsetError = xOffsetError;
  sim.detection.forwardError = forwardError;
  sim.detection.reprojError = reprojError;
  updateDetectionFromCurrentErrors();
  evaluateDetectionAcceptance();
  log(`${source} detection #${sim.detection.sampleIndex}: yaw ${yawError.toFixed(2)} deg, x ${xOffsetError.toFixed(3)} m, fwd ${forwardError.toFixed(3)} m, reproj ${reprojError.toFixed(2)} px`);
}

function updateDetectionFromCurrentErrors() {
  if (!sim.detection.active) return;
  if (sim.detection.source === "json") return;
  const t = palletTangent(sim.pallet);
  const n = palletNormal(sim.pallet);
  const c = {
    x: sim.pallet.x + t.x * sim.detection.xOffsetError - n.x * sim.detection.forwardError,
    y: sim.pallet.y + t.y * sim.detection.xOffsetError - n.y * sim.detection.forwardError
  };
  sim.detection.pallet = {
    x: c.x,
    y: c.y,
    yaw: wrapDeg(sim.pallet.yaw + sim.detection.yawError),
    width: sim.pallet.width,
    depth: sim.pallet.depth,
    height: sim.pallet.height
  };
}

function evaluateDetectionAcceptance() {
  sim.detection.inFov = isPalletInFov();
  sim.detection.accepted = sim.detection.inFov && sim.detection.reprojError <= sim.params.reprojAccept;
}

function isPalletInFov() {
  const camPos = cameraPosition();
  const camHeading = sim.fork.yaw;
  const dx = sim.pallet.x - camPos.x;
  const dy = sim.pallet.y - camPos.y;
  const dist = Math.hypot(dx, dy);
  if (dist < sim.camera.rangeMin || dist > sim.camera.rangeMax) return false;
  const bearing = angleDeg({ x: dx, y: dy });
  const offAxis = Math.abs(angleDelta(bearing, camHeading));
  return offAxis <= sim.camera.fovHorizontalDeg / 2;
}

function cameraPosition() {
  const h = headingVector(sim.fork.yaw);
  return {
    x: sim.fork.x + h.x * sim.camera.mountForward,
    y: sim.fork.y + h.y * sim.camera.mountForward
  };
}

function controlPallet() {
  return sim.detection.pallet || sim.pallet;
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
  const detectedPallet = controlPallet();
  const t = palletTangent(detectedPallet);
  const n = palletNormal(detectedPallet);
  const c = palletEntryFace(detectedPallet);
  const f = forkBodyCenter();
  const pPoint = forkPoint();
  const v = { x: f.x - c.x, y: f.y - c.y };
  const dForward = vectorDot(v, n);
  const dLateral = vectorDot(v, t);
  const h = headingVector(sim.fork.yaw);
  const targetPalletYaw = angleDeg({ x: -n.x, y: -n.y });

  const lateralDir = dLateral >= 0
    ? { x: -t.x, y: -t.y }
    : { x: t.x, y: t.y };
  const targetLateralYaw = angleDeg(lateralDir);
  const psiPallet = angleDelta(sim.fork.yaw, targetPalletYaw);
  const psiLateral = angleDelta(sim.fork.yaw, targetLateralYaw);
  const psiBodyAxis = angleDelta(angleDeg({ x: -h.y, y: h.x }), angleDeg(t));
  const foot = { x: c.x + n.x * dForward, y: c.y + n.y * dForward };

  sim.pose = {
    t, n, c, f, pPoint, foot, dForward, dLateral, psiPallet, psiLateral,
    psiBodyAxis, targetPalletYaw, targetLateralYaw
  };
  // DOPE-style 6D pose pipeline (pose6d_adapter.py reproduction).
  sim.dopePose = dopeStylePsi();
  return sim.pose;
}

// ===========================================================================
// Virtual 6D pose pipeline (DOPE-style ψ).
//
// 목적: 시뮬에서 depth_cam/calib/pose6d_adapter.py 의 ψ 변환 식을 실차와
// 동일하게 재현하여, FSM (align.py) 의 좌/우 회전 분기가 ψ 부호와 일치하는지
// 시각적으로 검증.
//
// 좌표계:
//   world: +x = canvas right, +y = canvas up, +z = up (시뮬 2D 에 가상의 z).
//   pallet model (v4): +X right(width), +Y up(height), +Z forward(entry face).
//   camera (OpenCV): +X right, +Y down, +Z forward (광축 = forklift heading).
// ===========================================================================

function matMul3(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
  return C;
}

function matT3(A) {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

function matVec3(A, v) {
  return [
    A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
    A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
    A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2],
  ];
}

// camera world pose. forklift heading 방향으로 +Z(광축), world -Z 가 카메라 +Y(down).
//   forward = (cos(yaw), sin(yaw), 0)
//   right   = (sin(yaw), -cos(yaw), 0)   // forward 의 시계방향 90°
//   down    = (0, 0, -1)
// R_world_camera = [right | down | forward] (컬럼).
function cameraWorldPose() {
  const yaw = sim.fork.yaw;
  const c = Math.cos(yaw * DEG);
  const s = Math.sin(yaw * DEG);
  // base: yaw 만 반영한 perfect forward-down 카메라 (perfect horizontal-forward)
  const R_wc_base = [
    [ s,  0,  c],
    [-c,  0,  s],
    [ 0, -1,  0],
  ];
  // body-frame tilt: R_tilt = R_z(roll) · R_x(pitch) (OpenCV camera frame)
  //   R_x(pitch) — pitch>0 → 광축이 아래로 (down-tilt)
  //   R_z(roll)  — roll>0  → viewer 기준 시계방향 회전
  // 마운트 적용: R_wc_final = R_wc_base · R_tilt (camera body 자체의 회전이라 right-multiply)
  const p = sim.camera.mountPitchDeg * DEG;
  const r = sim.camera.mountRollDeg  * DEG;
  const cp = Math.cos(p), sp = Math.sin(p);
  const cr = Math.cos(r), sr = Math.sin(r);
  const Rx = [[1,0,0],[0,cp,-sp],[0,sp,cp]];
  const Rz = [[cr,-sr,0],[sr,cr,0],[0,0,1]];
  const R_tilt = matMul3(Rz, Rx);
  const R_wc   = matMul3(R_wc_base, R_tilt);
  const cam = cameraPosition();
  const t_wc = [cam.x, cam.y, sim.camera.mountHeight];
  return { R: R_wc, t: t_wc };
}

// pallet world pose. pallet model +Z = palletNormal (entry face 방향, forklift 쪽).
//   +X (width)  = palletTangent
//   +Y (height) = world +z
//   +Z (depth)  = palletNormal
// R_world_pallet = [tangent | up | normal] (컬럼).
// t_world_pallet = (pallet.x, pallet.y, height/2).
function palletWorldPose() {
  const detectedPallet = controlPallet();
  const tan = palletTangent(detectedPallet);
  const nor = palletNormal(detectedPallet);
  const R_wp = [
    [tan.x, 0, nor.x],
    [tan.y, 0, nor.y],
    [0,     1, 0    ],
  ];
  const t_wp = [detectedPallet.x, detectedPallet.y, detectedPallet.height / 2];
  return { R: R_wp, t: t_wp };
}

// pallet pose in camera frame: R_cp = R_wc^T · R_wp, t_cp = R_wc^T · (t_wp - t_wc).
function palletInCameraFrame() {
  const { R: Rwc, t: twc } = cameraWorldPose();
  const { R: Rwp, t: twp } = palletWorldPose();
  const Rcw = matT3(Rwc);
  const Rcp = matMul3(Rcw, Rwp);
  const dt = [twp[0]-twc[0], twp[1]-twc[1], twp[2]-twc[2]];
  const tcp = matVec3(Rcw, dt);
  return { R: Rcp, t: tcp };
}

// pose6d_adapter.pose6d_to_align_vars 와 동일 식:
//   psi_rad = atan2(R[0][2], R[2][2])
//   psi_deg = degrees(psi_rad) + 180.0
//   psi_deg = wrap_to_180(psi_deg)
//   anchor = entry_face → (tx + R[0][2]*depth/2, tz + R[2][2]*depth/2)
function dopeStylePsi() {
  const { R, t } = palletInCameraFrame();
  let psi_deg = Math.atan2(R[0][2], R[2][2]) * RAD;
  psi_deg += 180.0;
  psi_deg = ((psi_deg + 180.0) % 360.0) - 180.0;
  if (psi_deg === -180.0) psi_deg = 180.0;
  const pallet = controlPallet();
  const offset = pallet.depth / 2.0;
  // DOPE camera frame: +X = camera right. align.py 의 d_lat 분기 식
  //   (mermaid: d_lat > tol → LATERAL_ROTATE_LEFT, d_lat < -tol → LATERAL_ROTATE_RIGHT)
  // 은 "forklift 가 pallet 의 우측 → 좌측 보정" 의도를 반영한다고 보면 부호가
  // 반전되어야 align.py 분기와 일치. 2026-05-27 실차 사용자 검증: forklift 가
  // 반대 방향 LATERAL chain 진입하던 문제 → 부호 반전으로 fix.
  const d_lat = -(t[0] + R[0][2] * offset);
  const d_fwd = t[2] + R[2][2] * offset;
  return { psi: psi_deg, dLat: d_lat, dFwd: d_fwd, R, t };
}

function update(dt) {
  syncParamsFromInputs();
  computePose();
  fsmTick(dt);
  constrainWorld();
  computePose();
  renderUi();
}

// =========================================================================
// PDF (그림 11) + 25y_automatic_lifter align.py snapshot-based FSM
// =========================================================================
// 흐름:
//   START → SEARCH (det_ok 대기)
//        → YAW_CHECK (snapshot psi, d_lat)
//             |psi| > tol            → YAW_CORRECT_RIGHT/LEFT → OFFSET_CHECK
//             |psi|≤tol ∧ |d_lat|≤tol → READY_TO_INSERT
//             |psi|≤tol ∧ |d_lat|>tol → OFFSET_CHECK
//        → OFFSET_CHECK (snapshot d_lat)
//             |d_lat| > tol → DIST_CHECK
//             else          → YAW_CHECK (loop)
//        → DIST_CHECK (snapshot d_lat, d_fwd)
//             |d_lat|>tol           → LATERAL chain (직교 정렬)
//             d_fwd-alignDist>band → ALIGN_FWD_ADJUST (timer)
//             alignDist-d_fwd>band → ALIGN_BWD_ADJUST (timer)
//             else                  → YAW_CHECK
//        → LATERAL chain: ROT (±90°) → FWD (t(|d_lat|)) → ROT_BACK (∓backYaw)
//        → READY_TO_INSERT → INSERT (timer (d_fwd+pocket)/slow) → DONE
// 시뮬 부호 매핑:
//   psi_pallet > 0 (forklift 가 target 보다 CCW 과회전) → ROT_RIGHT (yaw 감소)
//   psi_pallet < 0                                       → ROT_LEFT  (yaw 증가)
//   dLateral > 0 (forklift 가 entry face 의 +t 측) → LATERAL_ROTATE_RIGHT 먼저
//   dLateral < 0                                  → LATERAL_ROTATE_LEFT  먼저
// IMU rel_yaw 는 시뮬에서 fork.yaw 변화로 직접 추적.
// 전진 시간 t(d) = d / moveSpeed (단순화. PDF 의 piecewise 모델은 상수 미정).
// =========================================================================

function fsmTick(dt) {
  const f = sim.fsm;
  f.stateTime += dt;

  switch (f.state) {
    case "START":
      enterState("SEARCH");
      break;

    case "SEARCH":
      command("STOP");
      if (sim.detection.accepted) {
        enterState("YAW_CHECK");
        break;
      }
      if (f.stateTime > 0.5) {
        if (sim.detection.source === "json") {
          evaluateDetectionAcceptance();
        } else {
          sampleDetection("auto");
        }
        f.stateTime = 0;
        if (!sim.detection.accepted) log("[SEARCH] detection rejected (FOV/reproj)");
      }
      break;

    case "YAW_CHECK":         stepYawCheck();          break;
    case "YAW_CORRECT_RIGHT":
    case "YAW_CORRECT_LEFT":  stepYawCorrect(dt);      break;
    case "OFFSET_CHECK":      stepOffsetCheck();       break;
    case "DIST_CHECK":        stepDistCheck();         break;
    case "ALIGN_FWD_ADJUST":  stepFwdAdjust(dt, +1);   break;
    case "ALIGN_BWD_ADJUST":  stepFwdAdjust(dt, -1);   break;
    case "LATERAL_ROTATE_RIGHT":
    case "LATERAL_ROTATE_LEFT":
    case "FORWARD_AFTER_RIGHT":
    case "FORWARD_AFTER_LEFT":
    case "LATERAL_ROTATE_LEFT_BACK":
    case "LATERAL_ROTATE_RIGHT_BACK":
      stepLateralChain(dt);
      break;
    case "READY_TO_INSERT":   stepReadyToInsert();     break;
    case "INSERT":            stepInsert(dt);          break;

    case "DONE":
      command("STOP");
      break;

    case "STOP_INTERLOCK":
      command("STOP");
      timedState(dt, sim.params.stopSec, f.nextState || "SEARCH");
      break;

    default:
      enterState("START");
  }
}

// ----- snapshot ------------------------------------------------------------

function refreshPerceptionForSnapshot() {
  if (sim.detection.source === "json") {
    evaluateDetectionAcceptance();
  } else {
    sampleDetection("snapshot");
  }
  computePose();
}

function takeSnapshot() {
  refreshPerceptionForSnapshot();
  const p = sim.pose;
  // dopeFsm 토글 ON 이면 pose6d_adapter.py 변환 결과로 FSM 입력 대체.
  // 실차 ψ 부호 / d_lat 부호 일치 시각 검증용.
  let psi = p.psiPallet;
  let dLat = p.dLateral;
  let dFwd = p.dForward;
  if (sim.dopeFsm && sim.dopePose) {
    psi = sim.dopePose.psi;
    dLat = sim.dopePose.dLat;
    dFwd = sim.dopePose.dFwd;
  }
  sim.fsm.snapshot = {
    psi,
    dLateral: dLat,
    dForward: dFwd,
    ts: performance.now() / 1000
  };
  return sim.fsm.snapshot;
}

function relYawDelta(current, ref) {
  return wrapDeg(current - ref);
}

// motion_models.py 의 time_from_distance_piecewise 그대로.
//   d ≤ d_acc : t = t0 + sqrt(2d/a)
//   d >  d_acc: t = t0 + t1 + (d - d_acc)/vmax
//   clamp [fwdMinSec, fwdMaxSec].
function tDistanceSec(distM) {
  const d = Math.max(0, Math.abs(distM));
  const a = Math.max(1e-9, sim.params.fwdA);
  const t1 = sim.params.fwdT1;
  const t0 = sim.params.fwdT0;
  const dAcc = 0.5 * a * t1 * t1;
  const vmax = Math.max(1e-6, a * t1);
  let t;
  if (d <= dAcc) {
    t = t0 + Math.sqrt(2 * d / a);
  } else {
    t = t0 + t1 + (d - dAcc) / vmax;
  }
  return Math.max(sim.params.fwdMinSec, Math.min(sim.params.fwdMaxSec, t));
}

// FWD/BACK 명령 진행 중 가속+정속 dynamics. cmdElapsed 는 명령 시작 후 경과 시간.
//   tt = cmdElapsed - t0 (latency 보정)
//   tt < t1   : v = a * tt        (가속)
//   tt ≥ t1   : v = a * t1 = vmax (정속)
function fwdSpeedAt(cmdElapsedSec) {
  const tt = cmdElapsedSec - sim.params.fwdT0;
  if (tt <= 0) return 0;
  if (tt < sim.params.fwdT1) return sim.params.fwdA * tt;
  return sim.params.fwdA * sim.params.fwdT1;  // vmax
}

// ----- CHECK steps ---------------------------------------------------------

function stepYawCheck() {
  command("STOP");
  const snap = takeSnapshot();
  const yawTol = sim.params.psiTol;
  const offTol = sim.params.latTol;
  const psi = snap.psi;
  const dLat = snap.dLateral;

  if (Math.abs(psi) > yawTol) {
    const direction = psi > 0 ? -1 : +1;  // -1: ROT_RIGHT (yaw 감소), +1: ROT_LEFT
    const psiAbs = Math.max(sim.params.yawTurnMinDeg, Math.abs(psi));
    sim.fsm.yawPlan = { direction, yawAbsDeg: psiAbs, refYaw: sim.fork.yaw };
    enterState(direction < 0 ? "YAW_CORRECT_RIGHT" : "YAW_CORRECT_LEFT");
    log(`[YAW_CHECK→${sim.fsm.state}] psi_snap=${psi.toFixed(2)}° |psi|=${psiAbs.toFixed(2)}°`);
    return;
  }
  if (Math.abs(dLat) <= offTol) {
    enterState("READY_TO_INSERT");
    log(`[YAW_CHECK→READY_TO_INSERT] yaw OK ∧ offset OK (dLat=${dLat.toFixed(3)})`);
    return;
  }
  enterState("OFFSET_CHECK");
  log(`[YAW_CHECK→OFFSET_CHECK] yaw OK, offset 보정 필요 (dLat=${dLat.toFixed(3)})`);
}

function stepOffsetCheck() {
  command("STOP");
  const snap = takeSnapshot();
  const offTol = sim.params.latTol;
  if (Math.abs(snap.dLateral) > offTol) {
    enterState("DIST_CHECK");
    log(`[OFFSET_CHECK→DIST_CHECK] |dLat|=${Math.abs(snap.dLateral).toFixed(3)} > tol`);
  } else {
    enterState("YAW_CHECK");
    log(`[OFFSET_CHECK→YAW_CHECK] offset OK`);
  }
}

function stepDistCheck() {
  command("STOP");
  const snap = takeSnapshot();
  const dLat = snap.dLateral;
  const dFwd = snap.dForward;
  const offTol = sim.params.latTol;
  const alignDist = sim.params.alignDistM;
  const band = sim.params.alignBandM;

  if (Math.abs(dLat) > offTol) { enterOffsetNeedCorrect(); return; }

  const delta = dFwd - alignDist;
  if (delta > band) {
    sim.fsm.fwdPlan = { direction: +1, distanceM: delta, fwdSec: tDistanceSec(delta), deadlineTs: null };
    enterState("ALIGN_FWD_ADJUST");
    log(`[DIST_CHECK→ALIGN_FWD_ADJUST] dFwd=${dFwd.toFixed(3)} Δ=${delta.toFixed(3)} t=${sim.fsm.fwdPlan.fwdSec.toFixed(2)}s`);
    return;
  }
  if (-delta > band) {
    sim.fsm.fwdPlan = { direction: -1, distanceM: -delta, fwdSec: tDistanceSec(-delta), deadlineTs: null };
    enterState("ALIGN_BWD_ADJUST");
    log(`[DIST_CHECK→ALIGN_BWD_ADJUST] dFwd=${dFwd.toFixed(3)} Δ=${(-delta).toFixed(3)} t=${sim.fsm.fwdPlan.fwdSec.toFixed(2)}s`);
    return;
  }
  enterState("YAW_CHECK");
  log(`[DIST_CHECK→YAW_CHECK] band OK (Δ=${delta.toFixed(3)}m)`);
}

// ----- CMD steps (perception 무시, IMU/timer 만) ----------------------------

function stepYawCorrect(dt) {
  const plan = sim.fsm.yawPlan;
  if (!plan) { enterStop("YAW_CHECK"); return; }
  command(plan.direction < 0 ? "ROT_RIGHT" : "ROT_LEFT");
  sim.fork.yaw = wrapDeg(sim.fork.yaw + plan.direction * sim.params.rotSpeed * dt);
  const delta = relYawDelta(sim.fork.yaw, plan.refYaw);
  if (Math.abs(delta) >= plan.yawAbsDeg) {
    log(`[${sim.fsm.state}→OFFSET_CHECK] Δ=${delta.toFixed(2)}°/${(plan.direction * plan.yawAbsDeg).toFixed(2)}°`);
    sim.fsm.yawPlan = null;
    enterStop("OFFSET_CHECK");
  }
}

function stepFwdAdjust(dt, direction) {
  const plan = sim.fsm.fwdPlan;
  if (!plan || plan.direction !== direction) { enterStop("DIST_CHECK"); return; }
  command(direction > 0 ? "FWD" : "BACK");
  // 가속(0→vmax) → 정속 dynamics. config.py 의 FWD_A/FWD_T1/FWD_T0 그대로.
  const v = fwdSpeedAt(sim.fsm.stateTime);
  moveAlongHeading(direction * v * dt);
  if (plan.deadlineTs === null) plan.deadlineTs = sim.fsm.stateTime + plan.fwdSec;
  if (sim.fsm.stateTime >= plan.deadlineTs) {
    log(`[${sim.fsm.state}→DIST_CHECK] timer 완료 (목표 ${plan.distanceM.toFixed(3)}m)`);
    sim.fsm.fwdPlan = null;
    enterStop("DIST_CHECK");
  }
}

function enterOffsetNeedCorrect() {
  const snap = sim.fsm.snapshot;
  if (!snap) { enterStop("YAW_CHECK"); return; }
  const dLat = snap.dLateral;
  // 시뮬 birdview 부호 기준 (align.py 의 d_lat 부호와는 정반대):
  //   시뮬 dLat = (forklift - C) · t
  //   dLat > 0 → forklift 가 entry face 의 +t 측 → 좌측 평행이동 필요 → ROT_LEFT 먼저
  //              (정렬 heading=-n 에서 ROT_LEFT 90° → -t 방향 heading → FWD → -t 이동)
  //   dLat < 0 → forklift 가 -t 측 → 우측 이동 → ROT_RIGHT 먼저
  // (참고) align.py 는 d_lat<0 → ROT_RIGHT 인데, 이는 PDF 그림 11 패널 4 의
  //        "d_lat<0 → 제자리좌회전" 과 부호가 반대. FSM 폴더 코드 점검 필요.
  const direction = dLat > 0 ? +1 : -1;   // +1: ROT_LEFT 먼저, -1: ROT_RIGHT 먼저
  sim.fsm.lateralPlan = {
    direction,
    yawAbsDeg: 90.0,
    fwdSec: tDistanceSec(dLat),
    backYawDeg: sim.params.lateralBackYawDeg,
    refYaw: sim.fork.yaw,
    deadlineTs: null
  };
  if (direction < 0) {
    enterState("LATERAL_ROTATE_RIGHT");
    log(`[DIST_CHECK→LATERAL_ROTATE_RIGHT] dLat=${dLat.toFixed(3)}m t_fwd=${sim.fsm.lateralPlan.fwdSec.toFixed(2)}s`);
  } else {
    enterState("LATERAL_ROTATE_LEFT");
    log(`[DIST_CHECK→LATERAL_ROTATE_LEFT] dLat=${dLat.toFixed(3)}m t_fwd=${sim.fsm.lateralPlan.fwdSec.toFixed(2)}s`);
  }
}

function stepLateralChain(dt) {
  const plan = sim.fsm.lateralPlan;
  if (!plan) { enterStop("YAW_CHECK"); return; }
  const sub = sim.fsm.state;

  if (sub === "LATERAL_ROTATE_RIGHT" || sub === "LATERAL_ROTATE_LEFT") {
    const sign = sub === "LATERAL_ROTATE_RIGHT" ? -1 : +1;
    command(sign < 0 ? "ROT_RIGHT" : "ROT_LEFT");
    sim.fork.yaw = wrapDeg(sim.fork.yaw + sign * sim.params.rotSpeed * dt);
    const delta = relYawDelta(sim.fork.yaw, plan.refYaw);
    if (Math.abs(delta) >= plan.yawAbsDeg) {
      const nextSub = sub === "LATERAL_ROTATE_RIGHT" ? "FORWARD_AFTER_RIGHT" : "FORWARD_AFTER_LEFT";
      plan.deadlineTs = null;
      log(`[${sub}→${nextSub}] Δ=${delta.toFixed(2)}°`);
      enterStop(nextSub);
    }
    return;
  }
  if (sub === "FORWARD_AFTER_RIGHT" || sub === "FORWARD_AFTER_LEFT") {
    command("FWD");
    const v = fwdSpeedAt(sim.fsm.stateTime);
    moveAlongHeading(v * dt);
    if (plan.deadlineTs === null) plan.deadlineTs = sim.fsm.stateTime + plan.fwdSec;
    if (sim.fsm.stateTime >= plan.deadlineTs) {
      plan.refYaw = sim.fork.yaw;
      plan.deadlineTs = null;
      const nextSub = sub === "FORWARD_AFTER_RIGHT" ? "LATERAL_ROTATE_LEFT_BACK" : "LATERAL_ROTATE_RIGHT_BACK";
      log(`[${sub}→${nextSub}]`);
      enterStop(nextSub);
    }
    return;
  }
  if (sub === "LATERAL_ROTATE_LEFT_BACK" || sub === "LATERAL_ROTATE_RIGHT_BACK") {
    const sign = sub === "LATERAL_ROTATE_LEFT_BACK" ? +1 : -1;
    command(sign > 0 ? "ROT_LEFT" : "ROT_RIGHT");
    sim.fork.yaw = wrapDeg(sim.fork.yaw + sign * sim.params.rotSpeed * dt);
    const delta = relYawDelta(sim.fork.yaw, plan.refYaw);
    if (Math.abs(delta) >= plan.backYawDeg) {
      log(`[${sub}→YAW_CHECK] Δ=${delta.toFixed(2)}°`);
      sim.fsm.lateralPlan = null;
      enterStop("YAW_CHECK");
    }
    return;
  }
}

function stepReadyToInsert() {
  command("STOP");
  // body forward edge (forklift mast/fork carriage 의 전면) 가 pallet entry face
  // 에서 INSERT_BODY_SAFETY_M 만큼 떨어지는 곳까지 전진. 실제 forklift mast 가
  // pallet 정면에 부딪히지 않도록 보호.
  // bodyFrontOffset = fork center → body forward edge = centerToP - frontY
  //   centerToP = 0.72m, frontY (drawForklift) = 0.05m → bodyFrontOffset = 0.67m
  // 종료 시 fork tip = body + forkLen = entry face + (forkLen - bodySafety) = 안쪽 1.00m
  //   pallet depth 1.30m 의 77% 침투, back face 까지 0.30m. 안전.
  const snap = sim.fsm.snapshot || { dForward: sim.pose.dForward };
  const bodyFrontOffset = sim.fork.centerToP - 0.05;   // fork center → body forward edge (m)
  const bodySafety = sim.params.insertBodySafetyM ?? 0.05;
  const total = Math.max(0, snap.dForward - bodyFrontOffset - bodySafety);
  const sec = Math.max(sim.params.fwdMinSec, Math.min(sim.params.fwdMaxSec,
    total / Math.max(0.01, sim.params.slowSpeed)));
  sim.fsm.insertPlan = { fwdSec: sec, deadlineTs: null };
  enterState("INSERT");
  log(`[READY_TO_INSERT→INSERT] dFwd=${snap.dForward.toFixed(3)} - bodyOff=${bodyFrontOffset.toFixed(2)} - bodySafety=${bodySafety.toFixed(2)} → t=${sec.toFixed(2)}s`);
}

function stepInsert(dt) {
  const plan = sim.fsm.insertPlan;
  if (!plan) { enterState("DONE"); return; }
  // align.py INSERT: 단순 FWD 명령 + INSERT_FWD_MPS 정속 (FWD piecewise dynamics 와 별개).
  command("FWD");
  moveAlongHeading(sim.params.slowSpeed * dt);
  if (plan.deadlineTs === null) plan.deadlineTs = sim.fsm.stateTime + plan.fwdSec;
  if (sim.fsm.stateTime >= plan.deadlineTs) {
    log(`[INSERT→DONE] 포켓 삽입 완료`);
    sim.fsm.insertPlan = null;
    enterState("DONE");
  }
}

// ----- state utils ---------------------------------------------------------

function enterState(state) {
  if (sim.fsm.state === state) return;
  const prev = sim.fsm.state;
  sim.fsm.state = state;
  sim.fsm.stateTime = 0;
  sim.fsm.timer = 0;
  log(`${prev} → ${state}`);
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
  const err = sim.params.sideShift ? sim.pose.psiPallet : sim.pose.psiLateral;
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

function constrainWorld() {
  const hw = sim.world.width / 2;
  const hh = sim.world.height / 2;
  sim.fork.x = clamp(sim.fork.x, -hw, hw);
  sim.fork.y = clamp(sim.fork.y, -hh, hh);
  sim.pallet.x = clamp(sim.pallet.x, -hw, hw);
  sim.pallet.y = clamp(sim.pallet.y, -hh, hh);
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
  el.psiPallet.textContent = `${p.psiPallet.toFixed(2)} deg`;
  if (el.psiGeometric) el.psiGeometric.textContent = `${p.psiPallet.toFixed(2)} deg`;
  if (el.psiDope && sim.dopePose) el.psiDope.textContent = `${sim.dopePose.psi.toFixed(2)} deg`;
  if (el.dopeDLat && sim.dopePose) el.dopeDLat.textContent = `${sim.dopePose.dLat.toFixed(3)} m`;
  if (el.dopeDFwd && sim.dopePose) el.dopeDFwd.textContent = `${sim.dopePose.dFwd.toFixed(3)} m`;
  const detTag = sim.detection.source === "json" ? "JSON" : "noise";
  const acceptTag = sim.detection.accepted ? "OK" : (sim.detection.inFov ? "REJ" : "FOV");
  el.detReadout.textContent = `${detTag}/${acceptTag} yaw ${sim.detection.yawError.toFixed(2)} / x ${sim.detection.xOffsetError.toFixed(3)} / fwd ${sim.detection.forwardError.toFixed(3)} / reproj ${sim.detection.reprojError.toFixed(2)} px`;
  el.fovReadout.textContent = `${sim.detection.inFov ? "in FOV" : "out of FOV"} (${sim.camera.fovHorizontalDeg} deg, ${sim.camera.rangeMin}-${sim.camera.rangeMax} m, pitch ${sim.camera.mountPitchDeg.toFixed(1)}° roll ${sim.camera.mountRollDeg.toFixed(1)}°)`;
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
  drawCameraFov(ctx);
  drawPallet(ctx);
  drawDetectionEstimate(ctx);
  drawStateGeometry(ctx);
  drawForklift(ctx);
  drawLegend(ctx);
  renderUi();
  drawFsmDiagram();
}

// ============================================================================
// FSM diagram (25y_automatic_lifter/depth_cam/ui/diagram.py 의 JS 포팅).
// OUTER (SEARCH/ALIGN/INSERT/DONE) + ALIGN sub (snapshot 모델) 두 칼럼.
// 시뮬 state 를 diagram node 에 매핑하여 active 표시.
// ============================================================================

const FSM_OUTER_NODES = ["SEARCH", "ALIGN", "INSERT", "DONE"];

// ALIGN 칼럼 레벨별 노드. diagram.py 의 align_levels_ordered 와 동일 순서.
// INSERT 는 OUTER 에만 두고 ALIGN 칼럼에서는 제거 (POS key 중복 방지).
const FSM_ALIGN_LEVELS = [
  ["YAW_CHECK"],
  ["YAW_CORRECT_RIGHT", "YAW_CORRECT_LEFT"],
  ["OFFSET_CHECK"],
  ["DIST_CHECK"],
  ["ALIGN_FWD_ADJUST", "ALIGN_BWD_ADJUST"],
  ["OFFSET_NEED_CORRECT"],
  ["LATERAL_ROTATE_RIGHT", "LATERAL_ROTATE_LEFT"],
  ["FORWARD_AFTER_RIGHT", "FORWARD_AFTER_LEFT"],
  ["LATERAL_ROTATE_LEFT_BACK", "LATERAL_ROTATE_RIGHT_BACK"],
  ["READY_TO_DONE"]
];

// 노드 박스 안에 표시할 짧은 라벨 (긴 이름은 abbreviation).
const FSM_NODE_LABEL = {
  "YAW_CORRECT_RIGHT": "YAW_CORR_R",
  "YAW_CORRECT_LEFT":  "YAW_CORR_L",
  "ALIGN_FWD_ADJUST":  "FWD_ADJ",
  "ALIGN_BWD_ADJUST":  "BWD_ADJ",
  "OFFSET_NEED_CORRECT":       "OFF_NEED_CORR",
  "LATERAL_ROTATE_RIGHT":      "LAT_ROT_R",
  "LATERAL_ROTATE_LEFT":       "LAT_ROT_L",
  "FORWARD_AFTER_RIGHT":       "FWD_AFT_R",
  "FORWARD_AFTER_LEFT":        "FWD_AFT_L",
  "LATERAL_ROTATE_LEFT_BACK":  "LAT_ROT_L_B",
  "LATERAL_ROTATE_RIGHT_BACK": "LAT_ROT_R_B"
};

const FSM_EDGES = [
  // OUTER
  ["SEARCH", "ALIGN"], ["ALIGN", "INSERT"], ["INSERT", "DONE"],
  // ALIGN 진입
  ["ALIGN", "YAW_CHECK"],
  // YAW_CHECK → 3 branches
  ["YAW_CHECK", "YAW_CORRECT_RIGHT"], ["YAW_CHECK", "YAW_CORRECT_LEFT"], ["YAW_CHECK", "READY_TO_DONE"],
  // YAW_CORRECT → OFFSET_CHECK
  ["YAW_CORRECT_RIGHT", "OFFSET_CHECK"], ["YAW_CORRECT_LEFT", "OFFSET_CHECK"],
  // OFFSET_CHECK → DIST_CHECK
  ["OFFSET_CHECK", "DIST_CHECK"],
  // DIST_CHECK → 4 branches + loop
  ["DIST_CHECK", "ALIGN_FWD_ADJUST"], ["DIST_CHECK", "ALIGN_BWD_ADJUST"],
  ["DIST_CHECK", "LATERAL_ROTATE_RIGHT"], ["DIST_CHECK", "LATERAL_ROTATE_LEFT"],
  ["DIST_CHECK", "YAW_CHECK"],
  // ALIGN_*_ADJUST → DIST_CHECK
  ["ALIGN_FWD_ADJUST", "DIST_CHECK"], ["ALIGN_BWD_ADJUST", "DIST_CHECK"],
  // LATERAL chain (우/좌)
  ["LATERAL_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT"],
  ["FORWARD_AFTER_RIGHT", "LATERAL_ROTATE_LEFT_BACK"],
  ["LATERAL_ROTATE_LEFT_BACK", "YAW_CHECK"],
  ["LATERAL_ROTATE_LEFT", "FORWARD_AFTER_LEFT"],
  ["FORWARD_AFTER_LEFT", "LATERAL_ROTATE_RIGHT_BACK"],
  ["LATERAL_ROTATE_RIGHT_BACK", "YAW_CHECK"],
  // READY_TO_DONE → top INSERT
  ["READY_TO_DONE", "INSERT"]
];

// 시뮬 state → (outer, alignSub) 매핑.
function mapSimStateToDiagram() {
  let state = sim.fsm.state;
  if (state === "STOP_INTERLOCK") state = sim.fsm.nextState || state;
  // READY_TO_INSERT 는 align.py 의 READY_TO_DONE 에 해당.
  const alignSubMap = {
    "READY_TO_INSERT": "READY_TO_DONE"
  };
  if (state === "START" || state === "SEARCH") return { outer: "SEARCH", sub: null };
  if (state === "DONE") return { outer: "DONE", sub: null };
  if (state === "READY_TO_INSERT") return { outer: "INSERT", sub: "READY_TO_DONE" };
  if (state === "INSERT") return { outer: "INSERT", sub: null };
  // 모든 ALIGN sub
  return { outer: "ALIGN", sub: (alignSubMap[state] || state) };
}

function drawFsmDiagram() {
  const canvas = sim.diagramCanvas;
  if (!canvas) return;
  const ctx = sim.diagramCtx;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(320, Math.floor(rect.width * dpr));
  const H = Math.max(420, Math.floor(rect.height * dpr));
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }

  // 스타일 (diagram.py 와 비슷)
  const PANEL_BG = "#191919";
  const GROUP_BG = "#232323";
  const GROUP_BORDER = "#464646";
  const GROUP_ACTIVE = "#3cc8ff";
  const TEXT_DIM = "#a5a5a5";
  const TEXT_NORM = "#e6e6e6";
  const TEXT_TITLE = "#d2d2d2";
  const NODE_FILL = "#2a2a2a";
  const NODE_FILL_DIM = "#3c3c3c";
  const NODE_BORDER = "#5f5f5f";
  const EDGE_COLOR = "#6e6e6e";

  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(0, 0, W, H);

  // 타이틀 + 현재 상태
  const sFont = `${Math.round(13 * dpr)}px Arial`;
  ctx.fillStyle = TEXT_TITLE;
  ctx.font = `bold ${Math.round(15 * dpr)}px Arial`;
  ctx.textAlign = "center";
  ctx.fillText("Calibration FSM (Snapshot Model)", W / 2, 22 * dpr);

  // 레이아웃 상수 (canvas 폭에 맞춰 노드 크기 적응)
  // mermaid spec: 좌측 column = SEARCH, 가운데 = ALIGN composite, 우측 column = INSERT/DONE
  const PAD = 12 * dpr;
  const TITLE_H = 36 * dpr;
  const GROUP_PAD = 10 * dpr;
  const COL_GAP = 10 * dpr;
  const LEVEL_GAP = 14 * dpr;
  const totalInnerW = W - 2 * PAD - 2 * COL_GAP;
  const leftW  = totalInnerW * 0.16;
  const rightW = totalInnerW * 0.16;
  const alignW = totalInnerW - leftW - rightW;
  const xLeft  = PAD;
  const xAlign = xLeft + leftW + COL_GAP;
  const xRight = xAlign + alignW + COL_GAP;
  const topY = TITLE_H;
  const alignInnerW = alignW - 2 * GROUP_PAD;
  const nodeW = Math.min(160 * dpr, (alignInnerW - COL_GAP) / 2);
  const nodeH = 26 * dpr;
  const sideNodeW = Math.min(nodeW, Math.min(leftW, rightW) - 4 * dpr);

  // 레이아웃
  const POS = {};
  const startY = topY + GROUP_PAD + 22 * dpr;

  // 좌측 column — SEARCH (vertically centered with ALIGN)
  // 우측 column — INSERT (위), DONE (아래)
  // ALIGN 먼저 배치하여 box height 산출 후 좌/우 vertical 정렬.

  let ya = startY;
  for (const level of FSM_ALIGN_LEVELS) {
    const cols = level.length;
    const totalW = cols * nodeW + (cols - 1) * COL_GAP;
    const sx = xAlign + GROUP_PAD + Math.max(0, (alignInnerW - totalW) / 2);
    for (let i = 0; i < cols; i++) {
      const name = level[i];
      POS[name] = { x: sx + i * (nodeW + COL_GAP), y: ya, w: nodeW, h: nodeH };
    }
    ya += nodeH + LEVEL_GAP;
  }
  const alignBoxBottom = ya - LEVEL_GAP + GROUP_PAD;   // ALIGN 박스 아래 가장자리
  const alignBoxTop = topY;
  const alignBoxH = alignBoxBottom - alignBoxTop;

  // 좌측: SEARCH 노드 — ALIGN 박스 수직 중앙에 배치
  const cxLeft = xLeft + (leftW - sideNodeW) / 2;
  const ySearch = alignBoxTop + alignBoxH / 2 - nodeH / 2;
  POS["SEARCH"] = { x: cxLeft, y: ySearch, w: sideNodeW, h: nodeH };

  // 우측: INSERT (위), DONE (아래) — ALIGN 박스 수직 균등 배치
  const cxRight = xRight + (rightW - sideNodeW) / 2;
  const yInsert = alignBoxTop + alignBoxH * 0.40 - nodeH / 2;
  const yDone   = alignBoxTop + alignBoxH * 0.60 - nodeH / 2;
  POS["INSERT"] = { x: cxRight, y: yInsert, w: sideNodeW, h: nodeH };
  POS["DONE"]   = { x: cxRight, y: yDone,   w: sideNodeW, h: nodeH };

  // ALIGN 노드 ("OUTER" 상태의 ALIGN top-level) 는 mermaid 처럼 좌측 외부에 별도
  // 노드로 두지 않고, ALIGN composite 박스 자체가 그 역할을 함.
  // 다만 FSM_OUTER_NODES 안의 "ALIGN" 도 active 인식 위해 좌측 column 아래 작은
  // 영역에 배치 (활성 표시용).
  const yAlignLabel = alignBoxTop + alignBoxH * 0.30 - nodeH / 2;
  POS["ALIGN"] = { x: cxLeft, y: yAlignLabel, w: sideNodeW, h: nodeH };

  const gyEnd = Math.max(alignBoxBottom, yDone + nodeH + GROUP_PAD);

  // active 매핑
  const dm = mapSimStateToDiagram();
  const activeOuter = dm.outer;
  const activeSub = dm.sub;
  const inAlignGroup = (activeOuter === "ALIGN" || activeOuter === "INSERT");

  // 그룹 박스 — ALIGN composite 만
  const drawGroup = (gx, gw, gy_top, gy_bot, title, active) => {
    ctx.fillStyle = GROUP_BG;
    ctx.fillRect(gx, gy_top, gw, gy_bot - gy_top);
    ctx.strokeStyle = active ? GROUP_ACTIVE : GROUP_BORDER;
    ctx.lineWidth = 2 * dpr;
    ctx.strokeRect(gx, gy_top, gw, gy_bot - gy_top);
    ctx.fillStyle = TEXT_TITLE;
    ctx.font = `bold ${Math.round(13 * dpr)}px Arial`;
    ctx.textAlign = "center";
    ctx.fillText(title, gx + gw / 2, gy_top + 16 * dpr);
  };
  drawGroup(xAlign, alignW, alignBoxTop, alignBoxBottom, "ALIGN", inAlignGroup);

  // 엣지
  ctx.strokeStyle = EDGE_COLOR;
  ctx.lineWidth = 1 * dpr;
  ctx.fillStyle = EDGE_COLOR;
  const center = (n) => POS[n] && { x: POS[n].x + POS[n].w / 2, y: POS[n].y + POS[n].h / 2 };
  for (const [u, v] of FSM_EDGES) {
    const a = center(u), b = center(v);
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    // 화살표 머리
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    const head = 6 * dpr;
    ctx.beginPath();
    ctx.moveTo(b.x, b.y);
    ctx.lineTo(b.x - head * Math.cos(ang - 0.4), b.y - head * Math.sin(ang - 0.4));
    ctx.lineTo(b.x - head * Math.cos(ang + 0.4), b.y - head * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();
  }

  // 노드 (엣지 위에 덮어 그리기). 라벨이 박스 폭을 넘으면 폰트 자동 축소.
  const drawNode = (name, p, active, dim) => {
    ctx.fillStyle = dim ? NODE_FILL_DIM : NODE_FILL;
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.strokeStyle = active ? GROUP_ACTIVE : NODE_BORDER;
    ctx.lineWidth = active ? (2.5 * dpr) : (1.5 * dpr);
    ctx.strokeRect(p.x, p.y, p.w, p.h);
    ctx.fillStyle = active ? TEXT_NORM : (dim ? TEXT_DIM : TEXT_NORM);
    const label = FSM_NODE_LABEL[name] || name;
    let fontPx = 11;
    ctx.font = `${active ? "bold " : ""}${Math.round(fontPx * dpr)}px Arial`;
    const maxTextW = p.w - 8 * dpr;
    while (ctx.measureText(label).width > maxTextW && fontPx > 7) {
      fontPx -= 1;
      ctx.font = `${active ? "bold " : ""}${Math.round(fontPx * dpr)}px Arial`;
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, p.x + p.w / 2, p.y + p.h / 2 + 1 * dpr);
    ctx.textBaseline = "alphabetic";
  };
  for (const name of FSM_OUTER_NODES) {
    drawNode(name, POS[name], name === activeOuter, false);
  }
  for (const level of FSM_ALIGN_LEVELS) {
    for (const name of level) {
      const active = inAlignGroup && (name === activeSub);
      const dim = !inAlignGroup;
      drawNode(name, POS[name], active, dim);
    }
  }

  // 하단 상태 표시
  ctx.fillStyle = "#e1e1e1";
  ctx.font = `${Math.round(11 * dpr)}px Arial`;
  ctx.textAlign = "left";
  let statusLine = `state: ${sim.fsm.state}`;
  if (sim.fsm.state === "STOP_INTERLOCK") statusLine += ` (next: ${sim.fsm.nextState || "?"})`;
  if (inAlignGroup && activeSub) statusLine += `  •  align.sub: ${activeSub}`;
  ctx.fillText(statusLine, 14 * dpr, H - 10 * dpr);
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

function drawPallet(ctx) {
  const t = palletTangent(sim.pallet);
  const n = palletNormal(sim.pallet);
  const centroid = { x: sim.pallet.x, y: sim.pallet.y };
  const c = palletEntryFace(sim.pallet);
  const yaw = Math.atan2(t.y, t.x);
  drawRotatedRect(ctx, centroid, sim.pallet.width, sim.pallet.depth, yaw, "#caa472", "#5a3b18", 2.5);

  drawForkOpenings(ctx, centroid, yaw);

  const left = { x: c.x - t.x * sim.pallet.width / 2, y: c.y - t.y * sim.pallet.width / 2 };
  const right = { x: c.x + t.x * sim.pallet.width / 2, y: c.y + t.y * sim.pallet.width / 2 };
  ctx.save();
  ctx.strokeStyle = "#5a3b18";
  ctx.lineWidth = 4;
  line(ctx, worldToCanvas(left), worldToCanvas(right));
  drawDot(ctx, worldToCanvas(c), 6, "#5a3b18");
  label(ctx, "C", worldToCanvas({ x: c.x + t.x * 0.08, y: c.y + t.y * 0.08 }), "#5a3b18");
  drawDot(ctx, worldToCanvas(centroid), 5, "#ffffff", "#5a3b18");
  label(ctx, "G", worldToCanvas({ x: centroid.x + t.x * 0.08, y: centroid.y + t.y * 0.08 }), "#5a3b18");
  ctx.restore();
}

function drawForkOpenings(ctx, center, yawRad) {
  const c = worldToCanvas(center);
  const scale = c.scale;
  const w = sim.pallet.forkOpening * scale;
  const h = sim.pallet.depth * 0.78 * scale;
  const offset = (sim.fork.forkGap / 2) * scale;
  ctx.save();
  ctx.translate(c.x, c.y);
  ctx.rotate(-yawRad);
  ctx.fillStyle = "#2a1d0a";
  ctx.globalAlpha = 0.55;
  ctx.fillRect(-offset - w / 2, -h / 2, w, h);
  ctx.fillRect(offset - w / 2, -h / 2, w, h);
  ctx.restore();
}

function drawCameraFov(ctx) {
  if (!sim.camera.show) return;
  const camPos = cameraPosition();
  const heading = sim.fork.yaw;
  const half = sim.camera.fovHorizontalDeg / 2 * DEG;
  // pitch down-tilt 이면 ground plane 에 projected effective range 가 cos(pitch) 만큼 줄어듦
  const pitchRad = (sim.camera.mountPitchDeg ?? 0) * DEG;
  const rng = sim.camera.rangeMax * Math.cos(Math.max(0, pitchRad));
  const headingRad = heading * DEG;
  const a0 = headingRad - half;
  const a1 = headingRad + half;
  const p0 = worldToCanvas(camPos);
  const p1 = worldToCanvas({ x: camPos.x + Math.cos(a0) * rng, y: camPos.y + Math.sin(a0) * rng });
  const p2 = worldToCanvas({ x: camPos.x + Math.cos(a1) * rng, y: camPos.y + Math.sin(a1) * rng });
  ctx.save();
  ctx.fillStyle = sim.detection.inFov ? "rgba(60, 160, 230, 0.10)" : "rgba(200, 80, 80, 0.10)";
  ctx.strokeStyle = sim.detection.inFov ? "rgba(50, 130, 200, 0.55)" : "rgba(190, 70, 70, 0.55)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(p0.x, p0.y);
  ctx.lineTo(p1.x, p1.y);
  ctx.arc(p0.x, p0.y, Math.hypot(p1.x - p0.x, p1.y - p0.y), Math.atan2(p1.y - p0.y, p1.x - p0.x), Math.atan2(p2.y - p0.y, p2.x - p0.x));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  drawDot(ctx, p0, 4, "#3382c8");
  label(ctx, "cam", { x: p0.x + 4, y: p0.y - 4 }, "#3382c8");
  ctx.restore();
}

function drawDetectionEstimate(ctx) {
  if (!sim.detection.pallet) return;
  const pallet = sim.detection.pallet;
  const t = palletTangent(pallet);
  const n = palletNormal(pallet);
  const centroid = { x: pallet.x, y: pallet.y };
  const c = palletEntryFace(pallet);
  const left = { x: c.x - t.x * pallet.width / 2, y: c.y - t.y * pallet.width / 2 };
  const right = { x: c.x + t.x * pallet.width / 2, y: c.y + t.y * pallet.width / 2 };
  const normalEnd = { x: centroid.x + n.x * 2.0, y: centroid.y + n.y * 2.0 };

  const color = sim.detection.accepted ? "#8c5bd6" : "#c74343";
  ctx.save();
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  line(ctx, worldToCanvas(left), worldToCanvas(right));
  ctx.lineWidth = 1.5;
  line(ctx, worldToCanvas(centroid), worldToCanvas(normalEnd));
  ctx.setLineDash([]);
  drawDot(ctx, worldToCanvas(c), 5, color, "#ffffff");
  label(ctx, "C_hat", worldToCanvas({ x: c.x + t.x * 0.08, y: c.y + t.y * 0.08 }), color);
  drawDot(ctx, worldToCanvas(centroid), 4, "#ffffff", color);
  label(ctx, "G_hat", worldToCanvas({ x: centroid.x + t.x * 0.08, y: centroid.y + t.y * 0.08 }), color);
  ctx.restore();
}

function drawStateGeometry(ctx) {
  const p = sim.pose || computePose();
  const c = p.c;
  const foot = p.foot;
  const f = p.f;
  const pPoint = p.pPoint;
  const normalEnd = { x: c.x + p.n.x * 2.5, y: c.y + p.n.y * 2.5 };

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

  ctx.fillStyle = "#3382c8";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  const camLocalY = (sim.fork.centerToP - sim.camera.mountForward) * scale;
  ctx.beginPath();
  ctx.arc(0, camLocalY, 6 * (window.devicePixelRatio || 1), 0, TAU);
  ctx.fill();
  ctx.stroke();

  ctx.restore();

  ctx.save();
  drawDot(ctx, worldToCanvas(pWorld), 5, "#ffffff", "#111");
  label(ctx, "P", worldToCanvas({ x: pWorld.x + 0.08, y: pWorld.y + 0.08 }), "#111");
  drawDot(ctx, worldToCanvas(centerWorld), 5, "#111", "#ffffff");
  label(ctx, "O", worldToCanvas({ x: centerWorld.x + 0.08, y: centerWorld.y + 0.08 }), "#111");
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

function drawPsiArc(ctx, p) {
  const center = worldToCanvas(p.f);
  const radius = 54 * (window.devicePixelRatio || 1);
  const a0 = -sim.fork.yaw * DEG;
  const target = p.targetPalletYaw;
  const a1 = -target * DEG;
  ctx.save();
  ctx.strokeStyle = "#e57924";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, a0, a1, angleDelta(sim.fork.yaw, target) > 0);
  ctx.stroke();
  label(ctx, "psi_pallet", { x: center.x + radius * 0.44, y: center.y - radius * 0.62 }, "#e57924");
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
  roundRect(ctx, x, y, 226 * dpr, 168 * dpr, 8 * dpr);
  ctx.fill();
  ctx.stroke();
  ctx.font = `${12 * dpr}px Arial`;
  const rows = [
    ["G", "pallet centroid (DOPE x/y)", "#5a3b18"],
    ["C", "pallet entry face center", "#5a3b18"],
    ["H", "foot point", "#111"],
    ["O", "forklift center", "#111"],
    ["P", "fork inner midpoint", "#111"],
    ["cam", "RealSense mount", "#3382c8"],
    ["dashed", "DOPE detection", "#8c5bd6"],
    ["red", "d_forward", "#e34a4a"],
    ["blue", "d_lateral", "#2f6fed"],
    ["orange", "psi_pallet", "#e57924"]
  ];
  rows.forEach((row, i) => {
    ctx.fillStyle = row[2];
    ctx.fillText(`${row[0]}: ${row[1]}`, x + 14 * dpr, y + (23 + i * 14) * dpr);
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

function palletTangent(pallet = controlPallet()) {
  return headingVector(pallet.yaw);
}

function palletNormal(pallet = controlPallet()) {
  const t = palletTangent(pallet);
  return { x: t.y, y: -t.x };
}

// pallet.x / pallet.y 는 cuboid 무게중심(centroid) 좌표.
// FSM 거리 계산에 쓰는 entry face C 는 centroid + n*depth/2 로 유도.
function palletEntryFace(pallet = controlPallet()) {
  const n = palletNormal(pallet);
  return {
    x: pallet.x + n.x * pallet.depth / 2,
    y: pallet.y + n.y * pallet.depth / 2
  };
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
