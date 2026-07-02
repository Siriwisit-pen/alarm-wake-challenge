import {
  FilesetResolver,
  PoseLandmarker,
} from "../vendor/mediapipe/vision_bundle.mjs";

const MODEL_URL = new URL("../models/pose_landmarker_lite.task", import.meta.url).href;
const WASM_URL = new URL("../vendor/mediapipe/wasm", import.meta.url).href;
const STORAGE_KEY = "alarm-wake-challenge-v1";
const MAX_EXERCISES = 4;

const EXERCISE_CONFIG = {
  squat: {
    label: "Squat",
    targetLabel: "Target reps",
    progressLabel: "Reps",
    defaultTarget: 10,
    min: 1,
    max: 200,
    unit: "reps",
    hint:
      "Face the camera. Keep your full body visible, then move from standing to a clear squat and back up.",
  },
  "jumping-jack": {
    label: "Jumping jack",
    targetLabel: "Target reps",
    progressLabel: "Reps",
    defaultTarget: 20,
    min: 1,
    max: 300,
    unit: "reps",
    hint:
      "Face the camera. A rep counts when you open arms and legs, then return to a closed stance.",
  },
  "push-up": {
    label: "Push-up",
    targetLabel: "Target reps",
    progressLabel: "Reps",
    defaultTarget: 10,
    min: 1,
    max: 200,
    unit: "reps",
    hint:
      "Set the laptop to the side if possible. Keep shoulders, elbows, hips, knees, and ankles visible.",
  },
  plank: {
    label: "Plank",
    targetLabel: "Target seconds",
    progressLabel: "Seconds",
    defaultTarget: 30,
    min: 5,
    max: 600,
    unit: "sec",
    hint:
      "Use a side view. The timer advances only while shoulders, hips, and ankles form a straight line.",
  },
};

const SENSITIVITY = {
  relaxed: {
    visibility: 0.38,
    squatDownAngle: 125,
    squatUpAngle: 150,
    jackOpenRatio: 1.12,
    jackClosedRatio: 1.02,
    pushDownAngle: 108,
    pushUpAngle: 142,
    bodyLineAngle: 145,
  },
  normal: {
    visibility: 0.48,
    squatDownAngle: 115,
    squatUpAngle: 160,
    jackOpenRatio: 1.25,
    jackClosedRatio: 0.92,
    pushDownAngle: 96,
    pushUpAngle: 152,
    bodyLineAngle: 155,
  },
  strict: {
    visibility: 0.58,
    squatDownAngle: 105,
    squatUpAngle: 168,
    jackOpenRatio: 1.38,
    jackClosedRatio: 0.82,
    pushDownAngle: 86,
    pushUpAngle: 162,
    bodyLineAngle: 163,
  },
};

const CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 31],
  [28, 32],
];

const dom = {
  addExerciseButton: document.querySelector("#addExerciseButton"),
  alarmStatus: document.querySelector("#alarmStatus"),
  alarmTime: document.querySelector("#alarmTime"),
  appStatus: document.querySelector("#appStatus"),
  cameraButton: document.querySelector("#cameraButton"),
  challengePosition: document.querySelector("#challengePosition"),
  completionBadge: document.querySelector("#completionBadge"),
  currentExerciseName: document.querySelector("#currentExerciseName"),
  exerciseList: document.querySelector("#exerciseList"),
  formScore: document.querySelector("#formScore"),
  fpsReadout: document.querySelector("#fpsReadout"),
  hintText: document.querySelector("#hintText"),
  installButton: document.querySelector("#installButton"),
  modeBadge: document.querySelector("#modeBadge"),
  modelStatus: document.querySelector("#modelStatus"),
  offlineStatus: document.querySelector("#offlineStatus"),
  overlay: document.querySelector("#overlay"),
  phaseReadout: document.querySelector("#phaseReadout"),
  progressBar: document.querySelector("#progressBar"),
  progressLabel: document.querySelector("#progressLabel"),
  queueSummary: document.querySelector("#queueSummary"),
  repCount: document.querySelector("#repCount"),
  resetButton: document.querySelector("#resetButton"),
  ringingBanner: document.querySelector("#ringingBanner"),
  saveAlarmButton: document.querySelector("#saveAlarmButton"),
  sensitivitySelect: document.querySelector("#sensitivitySelect"),
  startNowButton: document.querySelector("#startNowButton"),
  targetCount: document.querySelector("#targetCount"),
  webcam: document.querySelector("#webcam"),
};

const ctx = dom.overlay.getContext("2d");

let poseLandmarker;
let webcamStream;
let animationFrameId = 0;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let fps = 0;
let cameraRunning = false;
let audioContext;
let alarmIntervalId = 0;

const appState = {
  alarmArmed: false,
  alarmDueAt: 0,
  alarmTimerId: 0,
  challenge: [],
  challengeActive: false,
  currentIndex: 0,
  deferredInstallPrompt: null,
  fromArmedAlarm: false,
  sensitivity: "relaxed",
  time: "07:00",
  wakeLock: null,
};

const detector = {
  exercise: "squat",
  phase: "idle",
  count: 0,
  target: 10,
  completed: false,
  lastRepAt: 0,
  downSince: 0,
  openSince: 0,
  validPlankMs: 0,
  lastPlankTick: 0,
};

init();

async function init() {
  loadSettings();
  wireControls();
  renderExerciseRows();
  renderQueue();
  resetDetectorForCurrentExercise("Loading pose model.");
  renderAlarmStatus();
  registerServiceWorker();
  updateOnlineStatus();

  if (appState.alarmArmed) {
    scheduleAlarm();
  }

  try {
    setModelStatus("Loading model", "");
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      numPoses: 1,
      runningMode: "VIDEO",
    });
    setModelStatus("Model ready", "ready");
    dom.appStatus.textContent = "Model ready.";
    dom.cameraButton.disabled = false;
    dom.startNowButton.disabled = false;
  } catch (error) {
    console.error("Model load failed", error);
    setModelStatus("Model failed", "error");
    dom.appStatus.textContent = "Pose model could not load.";
    dom.cameraButton.disabled = true;
    dom.startNowButton.disabled = true;
  }
}

function wireControls() {
  dom.cameraButton.disabled = true;
  dom.startNowButton.disabled = true;
  dom.alarmTime.value = appState.time;
  dom.sensitivitySelect.value = appState.sensitivity;

  dom.addExerciseButton.addEventListener("click", addExercise);
  dom.alarmTime.addEventListener("change", () => {
    appState.time = dom.alarmTime.value || "07:00";
    persistSettings();
    if (appState.alarmArmed) {
      scheduleAlarm();
    }
    renderAlarmStatus();
  });
  dom.cameraButton.addEventListener("click", toggleCamera);
  dom.exerciseList.addEventListener("change", handleExerciseListChange);
  dom.exerciseList.addEventListener("input", handleExerciseTargetInput);
  dom.exerciseList.addEventListener("click", handleExerciseListClick);
  dom.installButton.addEventListener("click", installApp);
  dom.resetButton.addEventListener("click", resetChallenge);
  dom.saveAlarmButton.addEventListener("click", toggleAlarmArm);
  dom.sensitivitySelect.addEventListener("change", () => {
    appState.sensitivity = dom.sensitivitySelect.value;
    persistSettings();
    resetDetectorForCurrentExercise("Sensitivity updated.");
  });
  dom.startNowButton.addEventListener("click", () => {
    startChallenge({ fromArmedAlarm: false });
  });

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    appState.deferredInstallPrompt = event;
    dom.installButton.hidden = false;
  });
  window.addEventListener("appinstalled", () => {
    dom.installButton.hidden = true;
    appState.deferredInstallPrompt = null;
  });
  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);
  window.addEventListener("resize", syncCanvasSize);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && appState.alarmArmed) {
      scheduleAlarm();
    }
  });
}

function loadSettings() {
  const fallback = {
    alarmArmed: false,
    challenge: [
      { exercise: "squat", target: 10 },
      { exercise: "jumping-jack", target: 15 },
    ],
    sensitivity: "relaxed",
    time: "07:00",
  };

  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    const settings = saved && typeof saved === "object" ? saved : fallback;
    appState.alarmArmed = Boolean(settings.alarmArmed);
    appState.challenge = normalizeChallenge(settings.challenge);
    appState.sensitivity = SENSITIVITY[settings.sensitivity]
      ? settings.sensitivity
      : fallback.sensitivity;
    appState.time = isValidTime(settings.time) ? settings.time : fallback.time;
  } catch {
    appState.alarmArmed = fallback.alarmArmed;
    appState.challenge = normalizeChallenge(fallback.challenge);
    appState.sensitivity = fallback.sensitivity;
    appState.time = fallback.time;
  }
}

function persistSettings() {
  const payload = {
    alarmArmed: appState.alarmArmed,
    challenge: normalizeChallenge(appState.challenge),
    sensitivity: appState.sensitivity,
    time: appState.time,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function normalizeChallenge(challenge) {
  const list = Array.isArray(challenge) ? challenge : [];
  const normalized = list
    .map((item) => {
      const exercise = EXERCISE_CONFIG[item?.exercise] ? item.exercise : "squat";
      const config = EXERCISE_CONFIG[exercise];
      const target = clamp(Number.parseInt(item?.target, 10), config.min, config.max);
      return {
        exercise,
        target: Number.isFinite(target) ? target : config.defaultTarget,
      };
    })
    .slice(0, MAX_EXERCISES);

  if (!normalized.length) {
    normalized.push({ exercise: "squat", target: EXERCISE_CONFIG.squat.defaultTarget });
  }
  return normalized;
}

function isValidTime(value) {
  return typeof value === "string" && /^\d{2}:\d{2}$/.test(value);
}

function addExercise() {
  if (appState.challenge.length >= MAX_EXERCISES) {
    return;
  }
  appState.challenge.push({
    exercise: "squat",
    target: EXERCISE_CONFIG.squat.defaultTarget,
  });
  persistSettings();
  renderExerciseRows();
  renderQueue();
  resetDetectorForCurrentExercise("Exercise added.");
}

function handleExerciseListChange(event) {
  const row = event.target.closest("[data-index]");
  if (!row || event.target.tagName !== "SELECT") {
    return;
  }

  const index = Number.parseInt(row.dataset.index, 10);
  const exercise = event.target.value;
  if (!EXERCISE_CONFIG[exercise]) {
    return;
  }

  const config = EXERCISE_CONFIG[exercise];
  appState.challenge[index] = {
    exercise,
    target: config.defaultTarget,
  };
  persistSettings();
  renderExerciseRows();
  renderQueue();
  if (index === appState.currentIndex) {
    resetDetectorForCurrentExercise("Exercise updated.");
  }
}

function handleExerciseTargetInput(event) {
  const row = event.target.closest("[data-index]");
  if (!row || event.target.type !== "number") {
    return;
  }

  const index = Number.parseInt(row.dataset.index, 10);
  const item = appState.challenge[index];
  if (!item) {
    return;
  }

  const config = EXERCISE_CONFIG[item.exercise];
  const parsed = Number.parseInt(event.target.value, 10);
  if (Number.isNaN(parsed)) {
    return;
  }
  item.target = clamp(parsed, config.min, config.max);
  persistSettings();
  renderQueue();
  if (index === appState.currentIndex) {
    resetDetectorForCurrentExercise("Target updated.");
  }
}

function handleExerciseListClick(event) {
  const button = event.target.closest("[data-remove]");
  if (!button) {
    return;
  }

  const index = Number.parseInt(button.dataset.remove, 10);
  if (appState.challenge.length <= 1) {
    return;
  }

  appState.challenge.splice(index, 1);
  appState.currentIndex = Math.min(appState.currentIndex, appState.challenge.length - 1);
  persistSettings();
  renderExerciseRows();
  renderQueue();
  resetDetectorForCurrentExercise("Exercise removed.");
}

function renderExerciseRows() {
  dom.exerciseList.innerHTML = "";

  for (const [index, item] of appState.challenge.entries()) {
    const config = EXERCISE_CONFIG[item.exercise];
    const row = document.createElement("div");
    row.className = "exercise-row";
    row.dataset.index = `${index}`;

    const exerciseField = document.createElement("div");
    exerciseField.className = "field-group";
    const exerciseLabel = document.createElement("label");
    exerciseLabel.textContent = `Exercise ${index + 1}`;
    const exerciseSelect = document.createElement("select");
    for (const [value, optionConfig] of Object.entries(EXERCISE_CONFIG)) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = optionConfig.label;
      option.selected = value === item.exercise;
      exerciseSelect.append(option);
    }
    exerciseField.append(exerciseLabel, exerciseSelect);

    const targetField = document.createElement("div");
    targetField.className = "field-group";
    const targetLabel = document.createElement("label");
    targetLabel.textContent = config.unit;
    const targetInput = document.createElement("input");
    targetInput.type = "number";
    targetInput.min = `${config.min}`;
    targetInput.max = `${config.max}`;
    targetInput.value = `${item.target}`;
    targetField.append(targetLabel, targetInput);

    const removeButton = document.createElement("button");
    removeButton.className = "icon-button";
    removeButton.type = "button";
    removeButton.textContent = "x";
    removeButton.title = "Remove exercise";
    removeButton.ariaLabel = "Remove exercise";
    removeButton.dataset.remove = `${index}`;
    removeButton.disabled = appState.challenge.length <= 1;

    row.append(exerciseField, targetField, removeButton);
    dom.exerciseList.append(row);
  }

  dom.addExerciseButton.disabled = appState.challenge.length >= MAX_EXERCISES;
}

function renderQueue() {
  dom.queueSummary.innerHTML = "";
  for (const [index, item] of appState.challenge.entries()) {
    const config = EXERCISE_CONFIG[item.exercise];
    const pill = document.createElement("span");
    pill.className = "queue-pill";
    pill.textContent = `${index + 1}. ${config.label} ${item.target} ${config.unit}`;
    dom.queueSummary.append(pill);
  }
}

function toggleAlarmArm() {
  if (appState.alarmArmed) {
    disarmAlarm();
    return;
  }

  armAlarm();
}

function armAlarm() {
  ensureAudioContext();
  appState.alarmArmed = true;
  appState.time = dom.alarmTime.value || "07:00";
  persistSettings();
  scheduleAlarm();
  renderAlarmStatus();
  dom.appStatus.textContent = "Alarm armed.";
}

function disarmAlarm() {
  clearTimeout(appState.alarmTimerId);
  appState.alarmTimerId = 0;
  appState.alarmDueAt = 0;
  appState.alarmArmed = false;
  persistSettings();
  renderAlarmStatus();
  dom.appStatus.textContent = "Alarm disarmed.";
}

function scheduleAlarm() {
  clearTimeout(appState.alarmTimerId);
  const dueAt = computeNextAlarm(appState.time);
  appState.alarmDueAt = dueAt.getTime();
  const delay = Math.max(0, appState.alarmDueAt - Date.now());
  appState.alarmTimerId = window.setTimeout(() => {
    triggerAlarm();
  }, delay);
  renderAlarmStatus();
}

function triggerAlarm() {
  appState.fromArmedAlarm = true;
  startChallenge({ fromArmedAlarm: true });
}

function computeNextAlarm(timeValue) {
  const [hours, minutes] = timeValue.split(":").map(Number);
  const now = new Date();
  const dueAt = new Date(now);
  dueAt.setHours(hours, minutes, 0, 0);
  if (dueAt.getTime() <= now.getTime() + 1000) {
    dueAt.setDate(dueAt.getDate() + 1);
  }
  return dueAt;
}

function renderAlarmStatus() {
  dom.saveAlarmButton.disabled = appState.challengeActive;
  dom.saveAlarmButton.textContent = appState.alarmArmed ? "Disarm" : "Arm alarm";

  if (!appState.alarmArmed) {
    dom.alarmStatus.textContent = "Not armed";
    return;
  }

  const dueAt = appState.alarmDueAt
    ? new Date(appState.alarmDueAt)
    : computeNextAlarm(appState.time);
  dom.alarmStatus.textContent = formatAlarmTime(dueAt);
}

function formatAlarmTime(date) {
  return date.toLocaleString([], {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function startChallenge({ fromArmedAlarm }) {
  if (!poseLandmarker) {
    dom.appStatus.textContent = "Pose model is still loading.";
    return;
  }

  appState.challenge = normalizeChallenge(appState.challenge);
  appState.challengeActive = true;
  appState.currentIndex = 0;
  appState.fromArmedAlarm = fromArmedAlarm;
  dom.completionBadge.hidden = true;
  dom.ringingBanner.hidden = false;
  renderAlarmStatus();
  setMode("ringing");
  resetDetectorForCurrentExercise("Alarm ringing.");
  startAlarmSound();
  requestWakeLock();
  startCamera();
}

function resetChallenge() {
  appState.currentIndex = 0;
  dom.completionBadge.hidden = true;
  if (appState.challengeActive) {
    dom.ringingBanner.hidden = false;
    setMode("ringing");
    resetDetectorForCurrentExercise("Challenge restarted.");
  } else {
    resetDetectorForCurrentExercise("Challenge reset.");
  }
}

function advanceChallenge() {
  if (appState.currentIndex < appState.challenge.length - 1) {
    appState.currentIndex += 1;
    resetDetectorForCurrentExercise("Next exercise.");
    renderQueue();
    return;
  }

  completeChallenge();
}

function completeChallenge() {
  appState.challengeActive = false;
  detector.completed = true;
  stopAlarmSound();
  releaseWakeLock();
  dom.ringingBanner.hidden = true;
  dom.completionBadge.textContent = "Alarm cleared";
  dom.completionBadge.hidden = false;
  dom.appStatus.textContent = "Alarm cleared.";
  setMode("complete");
  renderAlarmStatus();

  if (appState.fromArmedAlarm && appState.alarmArmed) {
    scheduleAlarm();
  }
}

async function toggleCamera() {
  if (cameraRunning) {
    stopCamera();
    return;
  }

  await startCamera();
}

async function startCamera() {
  if (!poseLandmarker || cameraRunning) {
    return;
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    dom.webcam.srcObject = webcamStream;
    await dom.webcam.play();
    syncCanvasSize();
    cameraRunning = true;
    dom.cameraButton.textContent = "Stop camera";
    dom.appStatus.textContent = appState.challengeActive
      ? "Tracking challenge."
      : "Camera tracking.";
    lastVideoTime = -1;
    lastFrameTime = performance.now();
    animationFrameId = requestAnimationFrame(predictWebcam);
  } catch (error) {
    console.error("Camera start failed", error);
    dom.appStatus.textContent = getCameraErrorMessage(error);
  }
}

function stopCamera() {
  cancelAnimationFrame(animationFrameId);
  animationFrameId = 0;
  cameraRunning = false;
  if (webcamStream) {
    for (const track of webcamStream.getTracks()) {
      track.stop();
    }
  }
  webcamStream = undefined;
  dom.webcam.srcObject = null;
  clearCanvas();
  dom.cameraButton.textContent = "Start camera";
  dom.appStatus.textContent = appState.challengeActive
    ? "Camera stopped. Alarm is still ringing."
    : "Camera is off.";
  dom.phaseReadout.textContent = "Phase: idle";
  dom.fpsReadout.textContent = "0 FPS";
}

function predictWebcam(now) {
  if (!cameraRunning) {
    return;
  }

  syncCanvasSize();

  if (dom.webcam.currentTime !== lastVideoTime) {
    lastVideoTime = dom.webcam.currentTime;
    const result = poseLandmarker.detectForVideo(dom.webcam, now);
    handlePoseResult(result, now);
    updateFps(now);
  }

  animationFrameId = requestAnimationFrame(predictWebcam);
}

function handlePoseResult(result, now) {
  clearCanvas();
  const landmarks = result.landmarks?.[0];

  if (!landmarks) {
    dom.formScore.textContent = "--";
    dom.appStatus.textContent = "No full body pose detected yet.";
    dom.phaseReadout.textContent = `Phase: ${detector.phase}`;
    return;
  }

  drawPose(landmarks);
  const thresholds = SENSITIVITY[appState.sensitivity];
  const indexBeforeEvaluation = appState.currentIndex;
  const evaluation = evaluateExercise(detector.exercise, landmarks, thresholds, now);

  dom.formScore.textContent = `${Math.round(evaluation.score)}%`;
  if (!detector.completed && appState.currentIndex === indexBeforeEvaluation) {
    dom.appStatus.textContent = evaluation.message;
  }
  dom.phaseReadout.textContent = `Phase: ${detector.phase}`;
  renderMetrics();
}

function evaluateExercise(exercise, landmarks, thresholds, now) {
  if (exercise === "squat") {
    return evaluateSquat(landmarks, thresholds, now);
  }
  if (exercise === "jumping-jack") {
    return evaluateJumpingJack(landmarks, thresholds, now);
  }
  if (exercise === "push-up") {
    return evaluatePushUp(landmarks, thresholds, now);
  }
  return evaluatePlank(landmarks, thresholds, now);
}

function evaluateSquat(landmarks, thresholds, now) {
  const needed = [23, 24, 25, 26, 27, 28];
  const visibility = averageVisibility(landmarks, needed);
  if (visibility < thresholds.visibility) {
    return lowVisibilityResult(visibility, thresholds);
  }

  const leftKnee = angle(landmarks[23], landmarks[25], landmarks[27]);
  const rightKnee = angle(landmarks[24], landmarks[26], landmarks[28]);
  const kneeAngle = averageNumbers([leftKnee, rightKnee]);
  const hipY = averageNumbers([landmarks[23].y, landmarks[24].y]);
  const kneeY = averageNumbers([landmarks[25].y, landmarks[26].y]);
  const standing = kneeAngle > thresholds.squatUpAngle && hipY < kneeY - 0.08;
  const down = kneeAngle < thresholds.squatDownAngle && hipY > kneeY - 0.02;

  if (detector.phase === "idle") {
    detector.phase = standing ? "up" : "finding up";
  }

  if (down && detector.phase !== "down") {
    detector.phase = "down";
    detector.downSince = now;
  }

  if (
    standing &&
    detector.phase === "down" &&
    now - detector.downSince > 220 &&
    canCount(now)
  ) {
    addRep(now);
    detector.phase = "up";
  } else if (standing) {
    detector.phase = "up";
  }

  return {
    message: down
      ? "Depth found. Stand back up to count the rep."
      : "Stand tall, then squat until your hips clearly drop.",
    score: scoreFromAngle(kneeAngle, thresholds.squatDownAngle, thresholds.squatUpAngle),
  };
}

function evaluateJumpingJack(landmarks, thresholds, now) {
  const needed = [11, 12, 15, 16, 27, 28];
  const visibility = averageVisibility(landmarks, needed);
  if (visibility < thresholds.visibility) {
    return lowVisibilityResult(visibility, thresholds);
  }

  const shoulderWidth = distance(landmarks[11], landmarks[12]);
  const ankleWidth = distance(landmarks[27], landmarks[28]);
  const wristsAboveShoulders =
    landmarks[15].y < landmarks[11].y && landmarks[16].y < landmarks[12].y;
  const wristsBelowShoulders =
    landmarks[15].y > landmarks[11].y + 0.08 &&
    landmarks[16].y > landmarks[12].y + 0.08;
  const open =
    ankleWidth > shoulderWidth * thresholds.jackOpenRatio && wristsAboveShoulders;
  const closed =
    ankleWidth < shoulderWidth * thresholds.jackClosedRatio && wristsBelowShoulders;

  if (detector.phase === "idle") {
    detector.phase = closed ? "closed" : "finding closed";
  }

  if (open && detector.phase !== "open") {
    detector.phase = "open";
    detector.openSince = now;
  }

  if (
    closed &&
    detector.phase === "open" &&
    now - detector.openSince > 180 &&
    canCount(now)
  ) {
    addRep(now);
    detector.phase = "closed";
  } else if (closed) {
    detector.phase = "closed";
  }

  return {
    message: open
      ? "Open position found. Return to closed stance to count."
      : "Open arms overhead and jump feet wider than shoulders.",
    score: clamp((ankleWidth / (shoulderWidth * thresholds.jackOpenRatio)) * 100, 0, 100),
  };
}

function evaluatePushUp(landmarks, thresholds, now) {
  const needed = [11, 12, 13, 14, 15, 16, 23, 24, 27, 28];
  const visibility = averageVisibility(landmarks, needed);
  if (visibility < thresholds.visibility) {
    return lowVisibilityResult(visibility, thresholds);
  }

  const leftElbow = angle(landmarks[11], landmarks[13], landmarks[15]);
  const rightElbow = angle(landmarks[12], landmarks[14], landmarks[16]);
  const elbowAngle = averageNumbers([leftElbow, rightElbow]);
  const leftLine = angle(landmarks[11], landmarks[23], landmarks[27]);
  const rightLine = angle(landmarks[12], landmarks[24], landmarks[28]);
  const bodyLine = averageNumbers([leftLine, rightLine]);
  const lineGood = bodyLine > thresholds.bodyLineAngle;
  const down = elbowAngle < thresholds.pushDownAngle && lineGood;
  const up = elbowAngle > thresholds.pushUpAngle && lineGood;

  if (detector.phase === "idle") {
    detector.phase = up ? "up" : "finding up";
  }

  if (down && detector.phase !== "down") {
    detector.phase = "down";
    detector.downSince = now;
  }

  if (
    up &&
    detector.phase === "down" &&
    now - detector.downSince > 220 &&
    canCount(now)
  ) {
    addRep(now);
    detector.phase = "up";
  } else if (up) {
    detector.phase = "up";
  }

  return {
    message: lineGood
      ? "Keep your body straight and bend elbows lower."
      : "Straighten your shoulder, hip, and ankle line before counting.",
    score: lineGood
      ? scoreFromAngle(elbowAngle, thresholds.pushDownAngle, thresholds.pushUpAngle)
      : clamp((bodyLine / 180) * 100, 0, 100),
  };
}

function evaluatePlank(landmarks, thresholds, now) {
  const needed = [11, 12, 23, 24, 27, 28];
  const visibility = averageVisibility(landmarks, needed);
  if (visibility < thresholds.visibility) {
    detector.lastPlankTick = now;
    detector.phase = "not visible";
    return lowVisibilityResult(visibility, thresholds);
  }

  const leftLine = angle(landmarks[11], landmarks[23], landmarks[27]);
  const rightLine = angle(landmarks[12], landmarks[24], landmarks[28]);
  const bodyLine = averageNumbers([leftLine, rightLine]);
  const valid = bodyLine > thresholds.bodyLineAngle;

  if (!detector.lastPlankTick) {
    detector.lastPlankTick = now;
  }

  if (valid && !detector.completed) {
    detector.validPlankMs += now - detector.lastPlankTick;
    detector.count = Math.floor(detector.validPlankMs / 1000);
    detector.phase = "holding";
    checkCompletion();
  } else if (!detector.completed) {
    detector.phase = "fix form";
  }
  detector.lastPlankTick = now;

  return {
    message: valid
      ? "Good plank line. Hold steady."
      : "Lift or lower your hips until shoulder, hip, and ankle line up.",
    score: clamp((bodyLine / 180) * 100, 0, 100),
  };
}

function addRep(now) {
  if (detector.completed) {
    return;
  }
  detector.count += 1;
  detector.lastRepAt = now;
  checkCompletion();
}

function canCount(now) {
  return !detector.completed && now - detector.lastRepAt > 650;
}

function checkCompletion() {
  if (detector.count < detector.target || detector.completed) {
    return;
  }

  detector.completed = true;
  if (appState.challengeActive) {
    advanceChallenge();
  } else {
    dom.completionBadge.textContent = "Exercise complete";
    dom.completionBadge.hidden = false;
    dom.appStatus.textContent = "Exercise complete.";
    setMode("complete");
  }
}

function resetDetectorForCurrentExercise(message) {
  const item = getCurrentChallengeItem();
  const config = EXERCISE_CONFIG[item.exercise];
  detector.exercise = item.exercise;
  detector.phase = "idle";
  detector.count = 0;
  detector.target = item.target;
  detector.completed = false;
  detector.lastRepAt = 0;
  detector.downSince = 0;
  detector.openSince = 0;
  detector.validPlankMs = 0;
  detector.lastPlankTick = 0;
  dom.completionBadge.hidden = true;
  dom.currentExerciseName.textContent = config.label;
  dom.challengePosition.textContent = `Exercise ${appState.currentIndex + 1} of ${appState.challenge.length}`;
  dom.hintText.textContent = config.hint;
  dom.progressLabel.textContent = config.progressLabel;
  dom.appStatus.textContent = message;
  dom.phaseReadout.textContent = "Phase: idle";
  renderMetrics();
}

function getCurrentChallengeItem() {
  appState.challenge = normalizeChallenge(appState.challenge);
  appState.currentIndex = clamp(appState.currentIndex, 0, appState.challenge.length - 1);
  return appState.challenge[appState.currentIndex];
}

function renderMetrics() {
  dom.repCount.textContent = `${Math.min(detector.count, detector.target)}`;
  dom.targetCount.textContent = `${detector.target}`;
  const progress = detector.target ? (detector.count / detector.target) * 100 : 0;
  dom.progressBar.style.width = `${clamp(progress, 0, 100)}%`;
}

function setModelStatus(text, className) {
  dom.modelStatus.textContent = text;
  dom.modelStatus.className = className;
}

function setMode(mode) {
  const labels = {
    setup: "Setup",
    ringing: "Ringing",
    complete: "Complete",
  };
  dom.modeBadge.textContent = labels[mode] || labels.setup;
  dom.modeBadge.className = `status-pill ${mode === "setup" ? "" : mode}`.trim();
}

async function ensureAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    dom.appStatus.textContent = "Alarm sound is not supported in this browser.";
    return null;
  }

  if (!audioContext) {
    audioContext = new AudioContextClass();
  }

  if (audioContext.state === "suspended") {
    try {
      await audioContext.resume();
    } catch (error) {
      console.warn("Audio resume failed", error);
    }
  }
  return audioContext;
}

async function startAlarmSound() {
  const context = await ensureAudioContext();
  if (!context || alarmIntervalId) {
    return;
  }

  playAlarmBeep(context);
  alarmIntervalId = window.setInterval(() => {
    playAlarmBeep(context);
  }, 720);
}

function playAlarmBeep(context) {
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  const now = context.currentTime;

  oscillator.type = "square";
  oscillator.frequency.setValueAtTime(880, now);
  oscillator.frequency.setValueAtTime(660, now + 0.16);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.18, now + 0.03);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.36);
  oscillator.connect(gain);
  gain.connect(context.destination);
  oscillator.start(now);
  oscillator.stop(now + 0.38);
}

function stopAlarmSound() {
  clearInterval(alarmIntervalId);
  alarmIntervalId = 0;
}

async function requestWakeLock() {
  if (!navigator.wakeLock?.request) {
    return;
  }

  try {
    appState.wakeLock = await navigator.wakeLock.request("screen");
  } catch (error) {
    console.warn("Wake lock failed", error);
  }
}

function releaseWakeLock() {
  if (!appState.wakeLock) {
    return;
  }

  appState.wakeLock.release();
  appState.wakeLock = null;
}

async function installApp() {
  if (!appState.deferredInstallPrompt) {
    dom.installButton.hidden = true;
    return;
  }

  appState.deferredInstallPrompt.prompt();
  await appState.deferredInstallPrompt.userChoice;
  appState.deferredInstallPrompt = null;
  dom.installButton.hidden = true;
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    dom.offlineStatus.textContent = "Offline cache unavailable.";
    return;
  }

  try {
    await navigator.serviceWorker.register("./sw.js");
    await navigator.serviceWorker.ready;
    dom.offlineStatus.textContent = "Offline ready.";
  } catch (error) {
    console.warn("Service worker failed", error);
    dom.offlineStatus.textContent = "Offline cache failed.";
  }
}

function updateOnlineStatus() {
  if (!navigator.onLine) {
    dom.offlineStatus.textContent = "Offline.";
  }
}

function getCameraErrorMessage(error) {
  if (!navigator.mediaDevices?.getUserMedia) {
    return "This browser does not expose webcam access. Try Chrome, Edge, or Safari.";
  }

  if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
    return "Camera permission was denied.";
  }

  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return "No webcam was found by the browser.";
  }

  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return "The webcam is busy or blocked by another app.";
  }

  if (error?.name === "OverconstrainedError") {
    return "The webcam could not match the requested settings.";
  }

  return `Camera failed: ${error?.name || "UnknownError"}.`;
}

function updateFps(now) {
  const delta = now - lastFrameTime;
  lastFrameTime = now;
  if (delta > 0) {
    fps = fps * 0.88 + (1000 / delta) * 0.12;
  }
  dom.fpsReadout.textContent = `${Math.round(fps)} FPS`;
}

function syncCanvasSize() {
  const width = dom.overlay.clientWidth;
  const height = dom.overlay.clientHeight;
  if (dom.overlay.width !== width || dom.overlay.height !== height) {
    dom.overlay.width = width;
    dom.overlay.height = height;
  }
}

function clearCanvas() {
  ctx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
}

function drawPose(landmarks) {
  const frame = getContainedVideoFrame();
  const points = landmarks.map((point) => ({
    ...point,
    x: frame.x + point.x * frame.width,
    y: frame.y + point.y * frame.height,
  }));

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(77, 213, 181, 0.92)";
  ctx.lineWidth = 4;

  for (const [start, end] of CONNECTIONS) {
    const a = points[start];
    const b = points[end];
    if (!a || !b) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(242, 193, 94, 0.95)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255, 115, 92, 0.95)";
    ctx.stroke();
  }
  ctx.restore();
}

function getContainedVideoFrame() {
  const canvasRatio = dom.overlay.width / dom.overlay.height;
  const videoRatio =
    dom.webcam.videoWidth && dom.webcam.videoHeight
      ? dom.webcam.videoWidth / dom.webcam.videoHeight
      : 16 / 9;

  if (canvasRatio > videoRatio) {
    const height = dom.overlay.height;
    const width = height * videoRatio;
    return {
      x: (dom.overlay.width - width) / 2,
      y: 0,
      width,
      height,
    };
  }

  const width = dom.overlay.width;
  const height = width / videoRatio;
  return {
    x: 0,
    y: (dom.overlay.height - height) / 2,
    width,
    height,
  };
}

function lowVisibilityResult(visibility, thresholds) {
  detector.phase = "not visible";
  return {
    message: "Move back or improve lighting so the full body is visible.",
    score: clamp((visibility / thresholds.visibility) * 100, 0, 100),
  };
}

function averageVisibility(landmarks, indexes) {
  const values = indexes.map((index) => landmarks[index]?.visibility ?? 0.75);
  return averageNumbers(values);
}

function averageNumbers(values) {
  const valid = values.filter((value) => Number.isFinite(value));
  if (!valid.length) {
    return 0;
  }
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angle(a, b, c) {
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let degrees = Math.abs((radians * 180) / Math.PI);
  if (degrees > 180) {
    degrees = 360 - degrees;
  }
  return degrees;
}

function scoreFromAngle(value, low, high) {
  const distanceFromTarget = Math.min(Math.abs(value - low), Math.abs(value - high));
  return clamp(100 - distanceFromTarget * 1.5, 0, 100);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
