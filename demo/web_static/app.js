"use strict";

const STAGES = {
  task1: [
    { id: "load", label: "Models" },
    { id: "input", label: "Image" },
    { id: "retrieve", label: "Candidates" },
    { id: "vlm", label: "VLM select" },
    { id: "output", label: "Output" },
  ],
  task2: [
    { id: "load", label: "Models" },
    { id: "input", label: "Image" },
    { id: "caption", label: "Captions" },
    { id: "rerank", label: "Scoring" },
    { id: "output", label: "Output" },
  ],
  calories: [
    { id: "load", label: "Models" },
    { id: "input", label: "Image" },
    { id: "composition", label: "Composition" },
    { id: "math", label: "Calories" },
    { id: "output", label: "Output" },
  ],
  both: [
    { id: "load", label: "Models" },
    { id: "input", label: "Image" },
    { id: "task1", label: "Task 1" },
    { id: "task2", label: "Task 2" },
    { id: "output", label: "Output" },
  ],
};

const TASK_LABELS = {
  task1: "Ingredients",
  task2: "Dish description",
  calories: "Calories",
  both: "Task 1 + Task 2",
};

const PRELOAD_MODELS = [
  { id: "siglip2", label: "SigLIP2" },
  { id: "qwen_instruct", label: "Qwen-VL-Instruct-4B" },
  { id: "qwen_reranker", label: "Qwen-VL-Reranker-2B" },
];

const PRELOAD_READY_STATES = new Set(["ready", "skipped"]);

const TASK_DEFAULT_SUBTITLES = {
  task1: "visual multi-label",
  task2: "caption alignment",
  calories: "composition estimate",
};

const VOICE_LISTEN_SECONDS = 2.5;

const PORTION_FACTORS = {
  none: 0,
  garnish: 0.25,
  small: 0.5,
  normal: 1,
  large: 2,
  double: 2,
};

const PORTION_OPTIONS = Object.keys(PORTION_FACTORS);
const COUNT_STEP = 0.5;

const els = {
  voiceButton: document.querySelector("#voiceButton"),
  voiceStatus: document.querySelector("#voiceStatus"),
  soundButton: document.querySelector("#soundButton"),
  soundIcon: document.querySelector("#soundIcon"),
  diagnosticsButton: document.querySelector("#diagnosticsButton"),
  floatingTooltip: document.querySelector("#floatingTooltip"),
  diagnosticsModal: document.querySelector("#diagnosticsModal"),
  diagnosticsCloseButton: document.querySelector("#diagnosticsCloseButton"),
  diagnosticsContent: document.querySelector("#diagnosticsContent"),
  cameraButton: document.querySelector("#cameraButton"),
  uploadButton: document.querySelector("#uploadButton"),
  prevImageButton: document.querySelector("#prevImageButton"),
  nextImageButton: document.querySelector("#nextImageButton"),
  imageUpload: document.querySelector("#imageUpload"),
  imageZone: document.querySelector("#imageZone"),
  imageFrame: document.querySelector("#imageFrame"),
  cameraVideo: document.querySelector("#cameraVideo"),
  imagePreview: document.querySelector("#imagePreview"),
  imageName: document.querySelector("#imageName"),
  imageState: document.querySelector("#imageState"),
  outputZone: document.querySelector(".output-zone"),
  answerText: document.querySelector("#answerText"),
  latencyMetric: document.querySelector("#latencyMetric"),
  outputMeta: document.querySelector("#outputMeta"),
  resultDetails: document.querySelector("#resultDetails"),
  pipelineSteps: document.querySelector("#pipelineSteps"),
  currentStage: document.querySelector("#currentStage"),
  toast: document.querySelector("#toast"),
  taskButtons: Array.from(document.querySelectorAll(".task-button[data-task]")),
  pipelineView: document.querySelector("#pipelineView"),
  personalView: document.querySelector("#personalView"),
  pipelineViewButton: document.querySelector("#pipelineViewButton"),
  personalViewButton: document.querySelector("#personalViewButton"),
  profileForm: document.querySelector("#profileForm"),
  profileStatus: document.querySelector("#profileStatus"),
  profileName: document.querySelector("#profileName"),
  profileSex: document.querySelector("#profileSex"),
  profileAge: document.querySelector("#profileAge"),
  profileHeight: document.querySelector("#profileHeight"),
  profileWeight: document.querySelector("#profileWeight"),
  profileActivity: document.querySelector("#profileActivity"),
  profileGoal: document.querySelector("#profileGoal"),
  budgetStatus: document.querySelector("#budgetStatus"),
  budgetCards: document.querySelector("#budgetCards"),
  requirementDetails: document.querySelector("#requirementDetails"),
  historyFrom: document.querySelector("#historyFrom"),
  historyTo: document.querySelector("#historyTo"),
  historyRefreshButton: document.querySelector("#historyRefreshButton"),
  historyDayViewButton: document.querySelector("#historyDayViewButton"),
  historyMonthViewButton: document.querySelector("#historyMonthViewButton"),
  historyForm: document.querySelector("#historyForm"),
  historyEditId: document.querySelector("#historyEditId"),
  historyComposerTitle: document.querySelector("#historyComposerTitle"),
  historyCalculatedCalories: document.querySelector("#historyCalculatedCalories"),
  historyConsumedAt: document.querySelector("#historyConsumedAt"),
  historyIngredients: document.querySelector("#historyIngredients"),
  addIngredientButton: document.querySelector("#addIngredientButton"),
  ingredientOptions: document.querySelector("#ingredientOptions"),
  historySubmitButton: document.querySelector("#historySubmitButton"),
  historyCancelEditButton: document.querySelector("#historyCancelEditButton"),
  historyGroups: document.querySelector("#historyGroups"),
  nutritionStats: document.querySelector("#nutritionStats"),
  nutritionChart: document.querySelector("#nutritionChart"),
};

const state = {
  selectedTask: "task1",
  imageData: "",
  imageName: "sample.jpg",
  sampleIndex: 0,
  sampleCount: 0,
  videoStream: null,
  busy: false,
  soundEnabled: true,
  ttsAudio: null,
  ttsObjectUrl: "",
  imageReadyTimer: null,
  voiceDeadline: 0,
  voiceTimer: null,
  voiceCapture: null,
  pollTimer: null,
  toastTimer: null,
  lastResult: null,
  lastJob: null,
  floatingTooltipAnchor: null,
  activeView: "pipeline",
  preload: {
    status: "idle",
    events: [],
    imageUnlocked: false,
    imageLoadStarted: false,
  },
  nutrition: {
    profile: null,
    requirement: null,
    summary: null,
    history: [],
    editIngredients: [],
    ingredientOptions: [],
    ingredientByName: new Map(),
    pendingSource: null,
    submitAttempted: false,
    historyView: "day",
    rangeFrom: "",
    rangeTo: "",
    loaded: false,
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  els.toast.textContent = message;
  els.toast.classList.add("visible");
  state.toastTimer = window.setTimeout(() => {
    els.toast.classList.remove("visible");
  }, 3200);
}

function buttonTooltipText(button) {
  return String(button?.dataset.tooltip || "").trim();
}

function positionFloatingTooltip() {
  if (!state.floatingTooltipAnchor || !els.floatingTooltip) return;
  const rect = state.floatingTooltipAnchor.getBoundingClientRect();
  const tooltip = els.floatingTooltip;
  const gap = 10;
  const viewportPadding = 16;
  const tooltipRect = tooltip.getBoundingClientRect();
  let left = rect.right - tooltipRect.width;
  left = Math.max(viewportPadding, Math.min(left, window.innerWidth - tooltipRect.width - viewportPadding));
  let top = rect.bottom + gap;
  if (top + tooltipRect.height > window.innerHeight - viewportPadding) {
    top = Math.max(viewportPadding, rect.top - tooltipRect.height - gap);
  }
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function showFloatingTooltip(button) {
  const text = buttonTooltipText(button);
  if (!text || !els.floatingTooltip) return;
  els.floatingTooltip.textContent = text;
  els.floatingTooltip.hidden = false;
  state.floatingTooltipAnchor = button;
  positionFloatingTooltip();
}

function hideFloatingTooltip() {
  if (!els.floatingTooltip) return;
  state.floatingTooltipAnchor = null;
  els.floatingTooltip.hidden = true;
}

function setVoiceStatus(message, mode = "idle") {
  els.voiceStatus.textContent = message;
  els.voiceButton.classList.toggle("listening", mode === "listening");
  if (mode === "listening") {
    els.voiceButton.querySelector(".task-index").textContent = "REC";
  } else {
    els.voiceButton.querySelector(".task-index").textContent = "VC";
  }
}

function clearVoiceTimer() {
  if (state.voiceTimer) {
    window.clearInterval(state.voiceTimer);
    state.voiceTimer = null;
  }
}

function voiceRemainingSeconds() {
  return Math.max(0, (state.voiceDeadline - Date.now()) / 1000);
}

function startVoiceCountdown() {
  clearVoiceTimer();
  state.voiceTimer = window.setInterval(() => {
    const remaining = voiceRemainingSeconds();
    if (remaining <= 0) {
      stopVoiceCapture();
      return;
    }
    setVoiceStatus(`recording ${remaining.toFixed(1)}s`, "listening");
  }, 180);
}

function mergeFloat32Chunks(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function downsampleBuffer(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) {
    return buffer;
  }
  const ratio = inputRate / outputRate;
  const outputLength = Math.round(buffer.length / ratio);
  const output = new Float32Array(outputLength);
  let inputOffset = 0;
  for (let outputOffset = 0; outputOffset < outputLength; outputOffset += 1) {
    const nextInputOffset = Math.round((outputOffset + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let index = inputOffset; index < nextInputOffset && index < buffer.length; index += 1) {
      sum += buffer[index];
      count += 1;
    }
    output[outputOffset] = count ? sum / count : 0;
    inputOffset = nextInputOffset;
  }
  return output;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index += 1, offset += 2) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

async function audioBlobToDataUrl(blob) {
  return blobToDataUrl(blob);
}

async function transcribeVoiceBlob(blob) {
  setVoiceStatus("transcribing");
  setAnswerVisible(true);
  els.answerText.textContent = "Transcribing the recorded command locally.";
  const audioData = await audioBlobToDataUrl(blob);
  const response = await fetch("/api/voice-command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_data: audioData }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Voice transcription failed");
  }
  setSelectedTask(payload.task || state.selectedTask);
  setVoiceStatus(`accepted: ${TASK_LABELS[payload.task] || payload.task}`);
  setAnswerVisible(true);
  els.answerText.textContent = `Heard: "${payload.transcript}". Running ${TASK_LABELS[payload.task] || payload.task}.`;
  startJob(payload.task || state.selectedTask, payload.transcript || "");
}

async function stopVoiceCapture() {
  const capture = state.voiceCapture;
  if (!capture || capture.stopping) {
    return;
  }
  capture.stopping = true;
  clearVoiceTimer();
  state.voiceDeadline = 0;
  setVoiceStatus("processing audio");
  try {
    capture.processor.disconnect();
    capture.source.disconnect();
  } catch (error) {
    console.warn(error);
  }
  capture.stream.getTracks().forEach((track) => track.stop());
  try {
    await capture.context.close();
  } catch (error) {
    console.warn(error);
  }
  state.voiceCapture = null;
  const samples = mergeFloat32Chunks(capture.chunks);
  if (samples.length < capture.sampleRate * 0.25) {
    setVoiceStatus("nothing recorded");
    setAnswerVisible(true);
    els.answerText.textContent = "No voice audio was recorded. Click Voice command and speak after it says recording.";
    return;
  }
  const downsampled = downsampleBuffer(samples, capture.sampleRate, 16000);
  const wavBlob = encodeWav(downsampled, 16000);
  try {
    await transcribeVoiceBlob(wavBlob);
  } catch (error) {
    setVoiceStatus("not understood");
    setAnswerVisible(true);
    els.answerText.textContent = error.message || "Voice command was not understood.";
    showToast(error.message || "Voice command failed");
  }
}

async function startVoiceCapture() {
  if (state.busy) {
    showToast("Wait for the current pipeline to finish");
    return;
  }
  if (state.voiceCapture) {
    stopVoiceCapture();
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Microphone unavailable in this browser");
    return;
  }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    showToast("Audio recording is unavailable in this browser");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
  } catch (error) {
    setVoiceStatus("mic blocked");
    showToast(error.message || "Microphone permission denied");
    return;
  }
  const context = new AudioContextClass();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    event.outputBuffer.getChannelData(0).fill(0);
  };
  source.connect(processor);
  processor.connect(context.destination);
  state.voiceCapture = {
    stream,
    context,
    source,
    processor,
    chunks,
    sampleRate: context.sampleRate,
    stopping: false,
  };
  state.voiceDeadline = Date.now() + VOICE_LISTEN_SECONDS * 1000;
  setVoiceStatus(`recording ${VOICE_LISTEN_SECONDS.toFixed(1)}s`, "listening");
  setAnswerVisible(true);
  els.answerText.textContent =
    "Recording now. Say: find ingredients, describe the dish, execute both, or estimate calories.";
  startVoiceCountdown();
}

function setSoundState(enabled) {
  state.soundEnabled = enabled;
  els.soundButton.classList.toggle("active", enabled);
  els.soundButton.setAttribute("aria-pressed", enabled ? "true" : "false");
  els.soundButton.setAttribute("aria-label", enabled ? "Sound output on" : "Sound output off");
  els.soundButton.dataset.tooltip = enabled ? "Sound output on" : "Sound output off";
  els.soundIcon.innerHTML = enabled ? "&#128266;" : "&#128263;";
  if (!enabled) {
    stopCurrentTts();
  }
}

function stopCurrentTts() {
  if (state.ttsAudio) {
    state.ttsAudio.pause();
    state.ttsAudio.removeAttribute("src");
    state.ttsAudio = null;
  }
  if (state.ttsObjectUrl) {
    URL.revokeObjectURL(state.ttsObjectUrl);
    state.ttsObjectUrl = "";
  }
  els.soundButton.classList.remove("speaking");
}

async function speakResult(text) {
  if (!state.soundEnabled || !text) {
    return;
  }
  stopCurrentTts();
  try {
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      let message = "TTS failed";
      try {
        const payload = await response.json();
        message = payload.error || message;
      } catch (error) {
        console.warn(error);
      }
      throw new Error(message);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    state.ttsAudio = audio;
    state.ttsObjectUrl = url;
    els.soundButton.classList.add("speaking");
    audio.onended = stopCurrentTts;
    audio.onerror = () => {
      stopCurrentTts();
      showToast("TTS audio playback failed");
    };
    await audio.play();
  } catch (error) {
    stopCurrentTts();
    showToast(error.message || "TTS failed");
  }
}

function preloadAllowsImage(status) {
  return PRELOAD_READY_STATES.has(String(status || "").toLowerCase());
}

function setModelState(states, id, status) {
  const current = states.get(id);
  if (!current) return;
  if (current.status === "ready" && status === "loading") return;
  current.status = status;
}

function modelStatesFromPreload(preload) {
  const states = new Map(PRELOAD_MODELS.map((model) => [model.id, { ...model, status: "pending" }]));
  const events = Array.isArray(preload?.events) ? preload.events : [];
  events.forEach((event) => {
    const message = String(event?.message || "").toLowerCase();
    if (!message) return;
    if (message.includes("loading siglip2") && (message.includes("qwen") || message.includes("vlm"))) {
      setModelState(states, "siglip2", "loading");
      setModelState(states, "qwen_instruct", "loading");
    }
    if (message.includes("siglip2") && (message.includes("qwen") || message.includes("vlm")) && message.includes("ready")) {
      setModelState(states, "siglip2", "ready");
      setModelState(states, "qwen_instruct", "ready");
    }
    if (message.includes("preparing siglip2 caption")) {
      setModelState(states, "siglip2", "loading");
    }
    if (message.includes("siglip2 caption") && message.includes("ready")) {
      setModelState(states, "siglip2", "ready");
    }
    if (message.includes("loading qwen-vl-reranker")) {
      setModelState(states, "qwen_reranker", "loading");
    }
    if (message.includes("qwen-vl-reranker") && message.includes("ready")) {
      setModelState(states, "qwen_reranker", "ready");
    }
    if (message.includes("calorie backend") && message.includes("preparing")) {
      setModelState(states, "siglip2", "loading");
      setModelState(states, "qwen_instruct", "loading");
    }
    if (message.includes("calorie backend") && message.includes("ready")) {
      setModelState(states, "siglip2", "ready");
      setModelState(states, "qwen_instruct", "ready");
    }
  });

  const status = String(preload?.status || "idle").toLowerCase();
  if (preloadAllowsImage(status)) {
    states.forEach((model) => {
      model.status = "ready";
    });
  }
  if (status === "error") {
    const loading = Array.from(states.values()).find((model) => model.status === "loading");
    const failed = loading || Array.from(states.values()).find((model) => model.status !== "ready");
    if (failed) failed.status = "error";
  }
  return PRELOAD_MODELS.map((model) => states.get(model.id) || { ...model, status: "pending" });
}

function preloadStateLabel(preload, modelStates) {
  const status = String(preload?.status || "idle").toLowerCase();
  if (status === "error") return "Preload error";
  if (preloadAllowsImage(status)) return "Models ready";
  const active = modelStates.filter((model) => model.status === "loading");
  if (active.length === 1) return `Loading ${active[0].label}`;
  if (active.length === 2) return `Loading ${active[0].label} and ${active[1].label}`;
  if (active.length > 2) {
    const labels = active.map((model) => model.label);
    return `Loading ${labels.slice(0, -1).join(", ")}, and ${labels[labels.length - 1]}`;
  }
  if (status === "running") return "Loading models";
  return "Preparing";
}

function preloadPipelineMessage() {
  return preloadStateLabel(
    { status: state.preload.status, events: state.preload.events },
    modelStatesFromPreload({ status: state.preload.status, events: state.preload.events }),
  );
}

function imagePipelineMessage() {
  if (!state.preload.imageUnlocked) return "Waiting for models";
  if (state.imageData) return "Image ready";
  if (state.preload.imageLoadStarted) return "Loading sample image";
  return "Waiting for image";
}

function setImageUnlocked(unlocked) {
  const isUnlocked = Boolean(unlocked);
  state.preload.imageUnlocked = isUnlocked;
  if (els.imageZone) els.imageZone.hidden = !isUnlocked;
  setBusy(state.busy);
}

function maybeLoadInitialImage() {
  if (!state.preload.imageUnlocked || state.preload.imageLoadStarted || state.imageData) return;
  state.preload.imageLoadStarted = true;
  loadDatasetImage(0);
}

function renderPreloadStatus(preload) {
  const next = preload || { status: "idle", events: [] };
  state.preload.status = String(next.status || "idle").toLowerCase();
  state.preload.events = Array.isArray(next.events) ? next.events : [];
  setImageUnlocked(preloadAllowsImage(state.preload.status));
  maybeLoadInitialImage();
  if (!state.busy) {
    renderPipeline(state.selectedTask, [], "idle");
  }
}

function setBusy(isBusy) {
  state.busy = isBusy;
  els.imageFrame.classList.toggle("running", isBusy);
  const controlsDisabled = isBusy || !state.preload.imageUnlocked;
  els.taskButtons.forEach((button) => {
    button.disabled = controlsDisabled;
  });
  els.voiceButton.disabled = controlsDisabled;
  els.cameraButton.disabled = controlsDisabled;
  els.uploadButton.disabled = controlsDisabled;
  els.prevImageButton.disabled = controlsDisabled;
  els.nextImageButton.disabled = controlsDisabled;
}

function setSelectedTask(task) {
  state.selectedTask = task;
  els.taskButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.task === task);
  });
  renderPipeline(task, [], "idle");
}

function renderPipeline(task, events = [], status = "idle") {
  const stages = STAGES[task] || STAGES.task1;
  const latest = events.length ? events[events.length - 1] : null;
  const messageByStage = new Map();
  events.forEach((event) => {
    if (event && event.stage) {
      messageByStage.set(event.stage, event.message || "");
    }
  });
  const preloadReady = preloadAllowsImage(state.preload.status);
  const preloadFailed = state.preload.status === "error";
  if (!messageByStage.has("load")) {
    messageByStage.set("load", preloadPipelineMessage());
  }
  if (!messageByStage.has("input")) {
    messageByStage.set("input", imagePipelineMessage());
  }

  let activeStage = latest ? latest.stage : "";
  if (activeStage === "run") {
    activeStage = stages[Math.min(2, stages.length - 1)].id;
  }
  if (activeStage === "done") {
    activeStage = "output";
  }
  if (!latest && !preloadReady) {
    activeStage = "load";
  } else if (!latest && preloadReady && !state.imageData) {
    activeStage = "input";
  }
  const activeIndex = stages.findIndex((stage) => stage.id === activeStage);
  const completed = status === "completed";
  const failed = status === "error";

  els.pipelineSteps.innerHTML = stages
    .map((stage, index) => {
      const isActive = !completed && !failed && stage.id === activeStage;
      let isDone = completed || (!failed && latest && activeIndex >= 0 && index < activeIndex);
      if (!failed && stage.id === "load" && preloadReady && !isActive) {
        isDone = true;
      }
      if (!failed && stage.id === "input" && state.imageData && !isActive) {
        isDone = true;
      }
      const isError =
        (preloadFailed && stage.id === "load") || (failed && (stage.id === activeStage || index === activeIndex));
      const classes = ["step"];
      if (isActive) classes.push("active");
      if (isDone) classes.push("done");
      if (isError) classes.push("error");
      if (stage.id === "load" && !preloadReady && !preloadFailed) classes.push("loading-models");
      const stageMessage = messageByStage.get(stage.id) || "";
      return `
        <div class="${classes.join(" ")}">
          <span class="step-label">${escapeHtml(stage.label)}</span>
          <span class="step-message">${escapeHtml(stageMessage)}</span>
        </div>
      `;
    })
    .join("");

  if (failed) {
    els.currentStage.textContent = "Error";
  } else if (completed) {
    els.currentStage.textContent = "Done";
  } else if (latest && latest.message) {
    els.currentStage.textContent = latest.message;
  } else if (!preloadReady) {
    els.currentStage.textContent = preloadPipelineMessage();
  } else if (!state.imageData) {
    els.currentStage.textContent = imagePipelineMessage();
  } else {
    els.currentStage.textContent = "Ready";
  }
}

function clearOutputForNewImage() {
  setAnswerVisible(false);
  els.answerText.textContent = "";
  setLatencyMetric(null);
  els.outputMeta.innerHTML = "";
  els.resultDetails.innerHTML = "";
  els.diagnosticsContent.innerHTML = "";
  els.diagnosticsButton.disabled = true;
  state.lastResult = null;
  state.lastJob = null;
  const task2Button = els.taskButtons.find((button) => button.dataset.task === "task2");
  const subtitle = task2Button?.querySelector(".task-subtitle");
  if (subtitle) subtitle.textContent = TASK_DEFAULT_SUBTITLES.task2;
  if (!els.diagnosticsModal.hidden) closeDiagnosticsModal();
}

function updateImagePreview(dataUrl, name, statusText) {
  clearOutputForNewImage();
  state.imageData = dataUrl;
  state.imageName = name || "image.jpg";
  els.imagePreview.src = dataUrl;
  els.imageFrame.classList.add("has-image");
  els.imageFrame.classList.remove("has-video");
  window.clearTimeout(state.imageReadyTimer);
  els.imageFrame.classList.remove("ready-pulse");
  void els.imageFrame.offsetWidth;
  els.imageFrame.classList.add("ready-pulse");
  state.imageReadyTimer = window.setTimeout(() => {
    els.imageFrame.classList.remove("ready-pulse");
  }, 1100);
  els.imageName.textContent = state.imageName;
  els.imageState.textContent = statusText || "Ready";
  if (!state.busy) {
    renderPipeline(state.selectedTask, [], "idle");
  }
}

async function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Could not read image"));
    reader.readAsDataURL(blob);
  });
}

async function loadDatasetImage(index = 0) {
  if (!state.preload.imageUnlocked) return;
  try {
    const response = await fetch(`/api/sample-image?index=${encodeURIComponent(index)}`, { cache: "no-store" });
    if (!response.ok) {
      showToast("No dataset sample image available");
      return;
    }
    const blob = await response.blob();
    const dataUrl = await blobToDataUrl(blob);
    const headerIndex = Number.parseInt(response.headers.get("X-Dishcovery-Image-Index") || "0", 10);
    const headerCount = Number.parseInt(response.headers.get("X-Dishcovery-Image-Count") || "0", 10);
    const headerName = response.headers.get("X-Dishcovery-Image-Name") || "dataset-sample.jpg";
    state.sampleIndex = Number.isFinite(headerIndex) ? headerIndex : 0;
    state.sampleCount = Number.isFinite(headerCount) ? headerCount : state.sampleCount;
    const position = state.sampleCount ? `${state.sampleIndex + 1}/${state.sampleCount}` : "Dataset";
    updateImagePreview(dataUrl, headerName, position);
  } catch (error) {
    showToast(error.message || "Could not load dataset image");
  }
}

async function startCamera() {
  if (!state.preload.imageUnlocked) {
    showToast("Models are still loading");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Camera unavailable in this browser");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    if (state.videoStream) {
      state.videoStream.getTracks().forEach((track) => track.stop());
    }
    state.videoStream = stream;
    els.cameraVideo.srcObject = stream;
    clearOutputForNewImage();
    state.imageData = "";
    els.imagePreview.removeAttribute("src");
    els.imageFrame.classList.add("has-video");
    els.imageFrame.classList.remove("has-image");
    els.imageName.textContent = "Camera";
    els.imageState.textContent = "Live";
  } catch (error) {
    showToast(error.message || "Camera failed");
  }
}

function captureLiveFrame() {
  const video = els.cameraVideo;
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    showToast("Camera frame unavailable");
    return "";
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    showToast("Camera frame unavailable");
    return "";
  }
  context.drawImage(video, 0, 0, width, height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
  updateImagePreview(dataUrl, "camera-frame.jpg", "Camera");
  return dataUrl;
}

function ensureImageData() {
  if (state.imageData) {
    return true;
  }
  if (state.videoStream) {
    return Boolean(captureLiveFrame());
  }
  showToast("Image required");
  return false;
}

function handleUpload(file) {
  if (!state.preload.imageUnlocked) {
    showToast("Models are still loading");
    return;
  }
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showToast("Upload an image file");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    updateImagePreview(String(reader.result || ""), file.name || "upload.jpg", "Uploaded");
  };
  reader.onerror = () => showToast("Upload failed");
  reader.readAsDataURL(file);
}

function taskPayload(task, transcript = "") {
  const payload = {
    image_data: state.imageData,
    image_name: state.imageName,
  };
  if (transcript) {
    payload.transcript = transcript;
  } else {
    payload.task = task;
  }
  return payload;
}

async function startJob(task, transcript = "") {
  if (state.busy) return;
  if (!state.preload.imageUnlocked) {
    showToast("Models are still loading");
    return;
  }
  if (!ensureImageData()) return;
  const requestedTask = transcript ? state.selectedTask : task;
  setSelectedTask(requestedTask || "task1");
  setBusy(true);
  setAnswerVisible(true);
  els.answerText.textContent = transcript
    ? `Understanding command: "${transcript}".`
    : "Running " + (TASK_LABELS[task] || "pipeline") + ".";
  setLatencyMetric(null);
  els.outputMeta.innerHTML = "";
  els.resultDetails.innerHTML = "";
  state.lastResult = null;
  state.lastJob = null;
  els.diagnosticsButton.disabled = true;
  renderPipeline(requestedTask || "task1", [{ stage: "input", message: "Submitting image" }], "running");
  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(taskPayload(task || state.selectedTask, transcript)),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Pipeline request failed");
    }
    if (payload.task) {
      setSelectedTask(payload.task);
      if (transcript) {
        setVoiceStatus(`accepted: ${TASK_LABELS[payload.task] || payload.task}`);
      }
    }
    pollJob(payload.id);
  } catch (error) {
    setBusy(false);
    if (transcript) {
      setVoiceStatus("not understood");
    }
    renderPipeline(requestedTask || "task1", [{ stage: "error", message: error.message }], "error");
    els.answerText.textContent = error.message;
    showToast(error.message);
  }
}

async function pollJob(jobId) {
  window.clearTimeout(state.pollTimer);
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.error || "Could not read job");
    }
    if (job.task && job.task !== state.selectedTask) {
      setSelectedTask(job.task);
    }
    state.lastJob = job;
    renderPipeline(job.task || state.selectedTask, job.events || [], job.status);
    if (job.status === "completed") {
      setBusy(false);
      renderResult(job.result || {});
      return;
    }
    if (job.status === "error") {
      setBusy(false);
      setAnswerVisible(true);
      els.answerText.textContent = job.error || "Pipeline failed";
      setLatencyMetric(null);
      els.outputMeta.innerHTML = "";
      els.resultDetails.innerHTML = "";
      showToast(job.error || "Pipeline failed");
      return;
    }
    state.pollTimer = window.setTimeout(() => pollJob(jobId), 450);
  } catch (error) {
    setBusy(false);
    setAnswerVisible(true);
    els.answerText.textContent = error.message;
    showToast(error.message);
  }
}

function pill(text, color = "") {
  return `<span class="pill ${color}">${escapeHtml(text)}</span>`;
}

function formatNumber(value, decimals = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "--";
  return value.toFixed(decimals);
}

function formatMetric(value, unit = "", decimals = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "--";
  return `${value.toFixed(decimals)}${unit}`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function localDateString(date = new Date()) {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
}

function localDatetimeValue(date = new Date()) {
  return `${localDateString(date)}T${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
}

function dateDaysAgo(days) {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return localDateString(date);
}

function setDefaultNutritionRange() {
  if (!state.nutrition.rangeFrom) state.nutrition.rangeFrom = dateDaysAgo(29);
  if (!state.nutrition.rangeTo) state.nutrition.rangeTo = localDateString();
  if (els.historyFrom && !els.historyFrom.value) els.historyFrom.value = state.nutrition.rangeFrom;
  if (els.historyTo && !els.historyTo.value) els.historyTo.value = state.nutrition.rangeTo;
  if (els.historyConsumedAt && !els.historyConsumedAt.value) {
    els.historyConsumedAt.value = localDatetimeValue();
  }
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function apiJson(path, options = {}) {
  const fetchOptions = {
    method: options.method || "GET",
    headers: {},
  };
  if (options.body !== undefined) {
    fetchOptions.headers["Content-Type"] = "application/json";
    fetchOptions.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, fetchOptions);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function setIngredientOptions(ingredients) {
  if (!Array.isArray(ingredients) || !ingredients.length) return;
  state.nutrition.ingredientOptions = ingredients;
  state.nutrition.ingredientByName = new Map(
    ingredients.map((item) => [String(item.name || "").trim().toLowerCase(), item]).filter(([name]) => Boolean(name)),
  );
  if (els.ingredientOptions) {
    els.ingredientOptions.innerHTML = ingredients
      .map((item) => `<option value="${escapeHtml(item.name || "")}"></option>`)
      .join("");
  }
}

function ingredientReference(name) {
  return state.nutrition.ingredientByName.get(String(name || "").trim().toLowerCase()) || null;
}

async function ensureIngredientOptionsLoaded() {
  if (state.nutrition.ingredientOptions.length) return;
  const payload = await apiJson("/api/nutrition/ingredients");
  setIngredientOptions(Array.isArray(payload.ingredients) ? payload.ingredients : []);
}

function setActiveView(view) {
  state.activeView = view;
  const personal = view === "personal";
  if (els.pipelineView) els.pipelineView.hidden = personal;
  if (els.personalView) els.personalView.hidden = !personal;
  els.pipelineViewButton?.classList.toggle("active", !personal);
  els.personalViewButton?.classList.toggle("active", personal);
  if (personal) {
    refreshNutrition();
  }
}

function nutritionQuery() {
  setDefaultNutritionRange();
  state.nutrition.rangeFrom = els.historyFrom?.value || state.nutrition.rangeFrom;
  state.nutrition.rangeTo = els.historyTo?.value || state.nutrition.rangeTo;
  return `from=${encodeURIComponent(state.nutrition.rangeFrom)}&to=${encodeURIComponent(state.nutrition.rangeTo)}`;
}

async function refreshNutrition(options = {}) {
  setDefaultNutritionRange();
  const query = nutritionQuery();
  try {
    const [profilePayload, summaryPayload, historyPayload, ingredientPayload] = await Promise.all([
      apiJson("/api/nutrition/profile"),
      apiJson(`/api/nutrition/summary?${query}`),
      apiJson(`/api/nutrition/history?${query}`),
      state.nutrition.ingredientOptions.length
        ? Promise.resolve({ ingredients: state.nutrition.ingredientOptions })
        : apiJson("/api/nutrition/ingredients"),
    ]);
    state.nutrition.profile = profilePayload.profile || {};
    state.nutrition.requirement = profilePayload.requirement || {};
    state.nutrition.summary = summaryPayload || {};
    state.nutrition.history = Array.isArray(historyPayload.entries) ? historyPayload.entries : [];
    setIngredientOptions(Array.isArray(ingredientPayload.ingredients) ? ingredientPayload.ingredients : []);
    state.nutrition.loaded = true;
    if (options.render !== false) {
      renderNutritionState();
      if (state.lastResult?.task === "calories") {
        els.resultDetails.innerHTML = renderDetails(state.lastResult);
      }
    }
  } catch (error) {
    showToast(error.message || "Nutrition data unavailable");
  }
}

function profilePayloadFromForm() {
  return {
    name: els.profileName?.value || "",
    sex: els.profileSex?.value || "",
    age: numberOrNull(els.profileAge?.value),
    height_cm: numberOrNull(els.profileHeight?.value),
    weight_kg: numberOrNull(els.profileWeight?.value),
    activity_level: els.profileActivity?.value || "sedentary",
    daily_calorie_goal_kcal: numberOrNull(els.profileGoal?.value),
  };
}

function hydrateProfileForm(profile) {
  if (!profile || !els.profileForm) return;
  els.profileName.value = profile.name || "";
  els.profileSex.value = profile.sex || "";
  els.profileAge.value = profile.age ?? "";
  els.profileHeight.value = profile.height_cm ?? "";
  els.profileWeight.value = profile.weight_kg ?? "";
  els.profileActivity.value = profile.activity_level || "sedentary";
  els.profileGoal.value = profile.daily_calorie_goal_kcal ?? "";
}

function kcalText(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "--";
  return `${Math.round(value)} kcal`;
}

function requirementSourceLabel(requirement) {
  if (!requirement) return "No limit";
  if (requirement.source === "manual_goal") return "Manual goal";
  if (requirement.source === "estimated_requirement") return "Estimated";
  return "Incomplete";
}

function missingProfileText(requirement) {
  const labels = {
    sex: "sex",
    age: "age",
    height_cm: "height",
    weight_kg: "weight",
    activity_level: "activity",
  };
  const missing = Array.isArray(requirement?.missing_fields) ? requirement.missing_fields : [];
  return missing.map((field) => labels[field] || field).join(", ");
}

function renderBudgetCards(summary, requirement) {
  const today = summary?.today || {};
  const limit = typeof today.limit_kcal === "number" ? today.limit_kcal : requirement?.daily_limit_kcal;
  const consumed = typeof today.total_kcal === "number" ? today.total_kcal : 0;
  const remaining = typeof limit === "number" ? limit - consumed : null;
  const status = typeof remaining === "number" ? (remaining < 0 ? "Over" : "Under") : "No limit";
  if (els.budgetStatus) {
    els.budgetStatus.textContent = status;
    els.budgetStatus.classList.toggle("over", status === "Over");
    els.budgetStatus.classList.toggle("under", status === "Under");
  }
  if (!els.budgetCards) return;
  const cards = [
    { label: "Daily limit", value: kcalText(limit) },
    { label: "Consumed", value: kcalText(consumed) },
    {
      label: "Remaining",
      value: typeof remaining === "number" ? kcalText(remaining) : "--",
      tone: typeof remaining === "number" ? (remaining < 0 ? "over" : "under") : "",
    },
  ];
  els.budgetCards.innerHTML = cards
    .map(
      (card) => `
        <div class="budget-card ${card.tone || ""}">
          <span>${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.value)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderRequirementDetails(profile, requirement) {
  if (!els.requirementDetails) return;
  const source = requirementSourceLabel(requirement);
  const missing = missingProfileText(requirement);
  const rows = [
    ["Source", source],
    ["BMR", kcalText(requirement?.bmr_kcal)],
    ["Maintenance", kcalText(requirement?.maintenance_kcal)],
  ];
  if (missing) rows.push(["Missing", missing]);
  els.requirementDetails.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="requirement-row">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");
  if (els.profileStatus) {
    els.profileStatus.textContent = profile?.updated_at ? "Saved" : "Not saved";
  }
}

function renderStats(summary) {
  if (!els.nutritionStats) return;
  const stats = summary?.stats || {};
  const items = [
    ["Tracked days", stats.tracked_days ?? 0],
    ["Under limit", stats.under_limit_days ?? 0],
    ["Over limit", stats.over_limit_days ?? 0],
    ["Daily avg", kcalText(stats.average_daily_kcal)],
  ];
  els.nutritionStats.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="stat-tile">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderNutritionChart(summary) {
  if (!els.nutritionChart) return;
  const days = Array.isArray(summary?.days) ? [...summary.days].reverse() : [];
  if (!days.length) {
    els.nutritionChart.innerHTML = `<p class="empty-state">No history in this range.</p>`;
    return;
  }
  const maxValue = Math.max(1, ...days.map((day) => day.total_kcal || 0), ...days.map((day) => day.limit_kcal || 0));
  els.nutritionChart.innerHTML = days
    .map((day) => {
      const totalPct = Math.max(1, Math.min(100, ((day.total_kcal || 0) / maxValue) * 100));
      const limitPct = typeof day.limit_kcal === "number" ? Math.max(1, Math.min(100, (day.limit_kcal / maxValue) * 100)) : 0;
      const tone = day.status === "over" ? "over" : "under";
      return `
        <div class="chart-row ${tone}">
          <span class="chart-date">${escapeHtml(day.date.slice(5))}</span>
          <div class="chart-track">
            ${limitPct ? `<i style="left:${limitPct.toFixed(1)}%"></i>` : ""}
            <b style="width:${totalPct.toFixed(1)}%"></b>
          </div>
          <strong>${escapeHtml(kcalText(day.total_kcal))}</strong>
        </div>
      `;
    })
    .join("");
}

function historyGroupLabel(key) {
  if (state.nutrition.historyView === "month") return key;
  const date = new Date(`${key}T00:00:00`);
  if (Number.isNaN(date.getTime())) return key;
  return date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

function formatEntryTime(value) {
  const text = String(value || "");
  if (text.length < 16) return text;
  return text.slice(11, 16);
}

function normalizePortion(value, fallback = "normal") {
  const portion = String(value || "").trim().toLowerCase();
  return PORTION_OPTIONS.includes(portion) ? portion : fallback;
}

function roundEditedKcal(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return 0;
  return Math.max(1, Math.floor(value + 0.5));
}

function firstNumeric(...values) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function countReferenceKcal(item) {
  const direct = firstNumeric(item.calories_per_single_object, item.per_instance_kcal);
  if (direct !== null && direct > 0) return direct;
  if (typeof item.base_kcal === "number" && typeof item.base_count === "number" && item.base_count > 0) {
    return item.base_kcal / item.base_count;
  }
  if (typeof item.kcal === "number" && typeof item.count === "number" && item.count > 0) {
    return item.kcal / item.count;
  }
  return null;
}

function portionReferenceKcal(item) {
  const direct = firstNumeric(item.calories_per_portion, item.average_portion_kcal);
  if (direct !== null && direct > 0) return direct;
  const single = firstNumeric(item.calories_per_single_object, item.per_instance_kcal);
  if (single !== null && single > 0) return single;
  const basePortion = normalizePortion(item.base_portion_category, "");
  const baseFactor = PORTION_FACTORS[basePortion];
  if (typeof item.base_kcal === "number" && baseFactor > 0) return item.base_kcal / baseFactor;
  const portion = normalizePortion(item.portion_category, "normal");
  const factor = PORTION_FACTORS[portion];
  if (typeof item.kcal === "number" && factor > 0) return item.kcal / factor;
  return null;
}

function ingredientWithReference(item) {
  const name = String(item?.name || "").trim().toLowerCase();
  const reference = ingredientReference(name);
  if (!reference) {
    return { ...item, name, invalid_name: Boolean(name) };
  }
  return {
    ...reference,
    ...item,
    name,
    invalid_name: false,
  };
}

function ingredientMode(item) {
  return typeof item?.count === "number" && Number.isFinite(item.count) && item.count > 0 ? "count" : "portion";
}

function recalculateIngredient(item) {
  const next = ingredientWithReference(item);
  if (next.invalid_name || !next.name) {
    next.kcal = null;
    return next;
  }
  const mode = ingredientMode(next);
  if (mode === "count") {
    const reference = countReferenceKcal(next);
    if (reference !== null) {
      next.kcal = roundEditedKcal(next.count * reference);
    }
    next.portion_category = "none";
    return next;
  }
  next.count = null;
  next.portion_category = normalizePortion(next.portion_category, "normal");
  if (next.abundance_scene) {
    const reference = firstNumeric(next.per_instance_kcal, next.calories_per_single_object, next.kcal);
    if (reference !== null) {
      next.kcal = roundEditedKcal(reference);
    }
    return next;
  }
  const reference = portionReferenceKcal(next);
  if (reference !== null) {
    next.kcal = roundEditedKcal(PORTION_FACTORS[next.portion_category] * reference);
  }
  next.portion_factor = PORTION_FACTORS[next.portion_category];
  return next;
}

function ingredientForEditor(item) {
  const next = recalculateIngredient({ ...item });
  return {
    ...next,
    base_kcal: typeof next.kcal === "number" ? next.kcal : null,
    base_count: typeof next.count === "number" ? next.count : null,
    base_portion_category: next.portion_category || "",
  };
}

function formatIngredientLine(item) {
  const name = String(item?.name || "").trim();
  if (!name) return "";
  const quantity =
    item?.abundance_scene && ingredientMode(item) !== "count"
      ? "portion: one item"
      : ingredientMode(item) === "count"
      ? `count: ${formatCount(item.count)}`
      : `portion: ${normalizePortion(item.portion_category, "normal")}`;
  const kcal = typeof item.kcal === "number" && Number.isFinite(item.kcal) ? ` · ${Math.round(item.kcal)} kcal` : "";
  return `${name} · ${quantity}${kcal}`;
}

function renderIngredientEditor() {
  if (!els.historyIngredients) return;
  const ingredients = state.nutrition.editIngredients || [];
  if (!ingredients.length) {
    els.historyIngredients.innerHTML = `<p class="empty-state compact">No ingredient details for this dish.</p>`;
    updateHistoryCalculatedCalories(0);
    return;
  }
  els.historyIngredients.innerHTML = ingredients
    .map((rawItem, index) => {
      const item = ingredientWithReference(rawItem);
      const name = String(item.name || "");
      const mode = ingredientMode(item);
      const countRef = countReferenceKcal(item);
      const portionRef = portionReferenceKcal(item);
      const countAllowed = countRef !== null;
      const portionAllowed = portionRef !== null;
      const portion = normalizePortion(item.portion_category, "normal");
      const count = typeof item.count === "number" && Number.isFinite(item.count) ? item.count : 1;
      const hasTypedName = Boolean(name);
      const invalidName = Boolean((hasTypedName && !ingredientReference(name)) || (state.nutrition.submitAttempted && !hasTypedName));
      return `
        <div class="ingredient-edit-row ${invalidName ? "invalid" : ""}" data-ingredient-index="${index}">
          <label class="ingredient-name-field">
            <span>Ingredient</span>
            <input
              data-ingredient-control="name"
              type="text"
              list="ingredientOptions"
              value="${escapeHtml(name)}"
              placeholder="ingredient from list"
              required
            />
          </label>
          <label>
            <span>Keyword</span>
            <select data-ingredient-control="mode">
              <option value="portion" ${mode === "portion" ? "selected" : ""} ${portionAllowed ? "" : "disabled"}>Portion</option>
              <option value="count" ${mode === "count" ? "selected" : ""} ${countAllowed ? "" : "disabled"}>Count</option>
            </select>
          </label>
          ${
            mode === "count"
              ? `<label>
                  <span>Count</span>
                  <input data-ingredient-control="count" type="number" min="0.5" max="100" step="0.5" value="${escapeHtml(count)}" required />
                </label>`
              : `<label>
                  <span>Portion</span>
                  <select data-ingredient-control="portion">
                    ${PORTION_OPTIONS.map(
                      (option) => `<option value="${escapeHtml(option)}" ${option === portion ? "selected" : ""}>${escapeHtml(option)}</option>`,
                    ).join("")}
                  </select>
                </label>`
          }
          <div class="ingredient-edit-meta">
            <span data-ingredient-kcal>${escapeHtml(kcalText(item.kcal))}</span>
            <button type="button" data-ingredient-control="remove" aria-label="Remove ingredient">Remove</button>
            ${invalidName ? `<em>Choose an ingredient from the list.</em>` : ""}
          </div>
        </div>
      `;
    })
    .join("");
  recalculateHistoryCaloriesFromIngredients();
}

function updateHistoryCalculatedCalories(total) {
  if (!els.historyCalculatedCalories) return;
  els.historyCalculatedCalories.textContent = total > 0 ? kcalText(total) : "--";
}

function syncIngredientFromRow(row) {
  const index = Number(row?.dataset?.ingredientIndex);
  if (!Number.isInteger(index) || !state.nutrition.editIngredients[index]) return;
  const item = { ...state.nutrition.editIngredients[index] };
  const rawName = row.querySelector('[data-ingredient-control="name"]')?.value || "";
  item.name = String(rawName).trim().toLowerCase();
  const mode = row.querySelector('[data-ingredient-control="mode"]')?.value || ingredientMode(item);
  if (mode === "count") {
    const countValue = numberOrNull(row.querySelector('[data-ingredient-control="count"]')?.value);
    item.count = countValue ?? (typeof item.count === "number" && item.count > 0 ? item.count : 1);
    item.portion_category = "none";
  } else {
    item.count = null;
    item.portion_category = normalizePortion(row.querySelector('[data-ingredient-control="portion"]')?.value, "normal");
  }
  state.nutrition.editIngredients[index] = recalculateIngredient(item);
}

function recalculateHistoryCaloriesFromIngredients() {
  const rows = Array.from(els.historyIngredients?.querySelectorAll(".ingredient-edit-row") || []);
  rows.forEach(syncIngredientFromRow);
  const total = state.nutrition.editIngredients.reduce((sum, item) => sum + Number(item.kcal || 0), 0);
  updateHistoryCalculatedCalories(Math.round(total));
  rows.forEach((row) => {
    const index = Number(row.dataset.ingredientIndex);
    const value = state.nutrition.editIngredients[index]?.kcal;
    const label = row.querySelector("[data-ingredient-kcal]");
    if (label) label.textContent = kcalText(value);
  });
}

function addManualIngredient() {
  state.nutrition.submitAttempted = false;
  state.nutrition.editIngredients.push({
    name: "",
    count: null,
    portion_category: "normal",
    kcal: null,
  });
  renderIngredientEditor();
}

function removeIngredientAt(index) {
  if (!Number.isInteger(index)) return;
  state.nutrition.submitAttempted = false;
  state.nutrition.editIngredients.splice(index, 1);
  renderIngredientEditor();
}

function validateHistoryIngredients() {
  recalculateHistoryCaloriesFromIngredients();
  state.nutrition.submitAttempted = true;
  if (!state.nutrition.editIngredients.length) {
    renderIngredientEditor();
    return "Add at least one ingredient from the list.";
  }
  for (const item of state.nutrition.editIngredients) {
    const name = String(item.name || "").trim().toLowerCase();
    if (!name || !ingredientReference(name)) {
      renderIngredientEditor();
      return "Choose each ingredient from the suggestion list.";
    }
    if (ingredientMode(item) === "count") {
      const count = numberOrNull(item.count);
      if (count === null || count < COUNT_STEP || Math.abs(count / COUNT_STEP - Math.round(count / COUNT_STEP)) > 1e-6) {
        return `Count must use ${COUNT_STEP} steps.`;
      }
    } else if (!PORTION_OPTIONS.includes(normalizePortion(item.portion_category, ""))) {
      return `Portion must be one of: ${PORTION_OPTIONS.join(", ")}.`;
    }
  }
  return "";
}

function generatedDishName() {
  const names = state.nutrition.editIngredients
    .map((item) => String(item.name || "").trim())
    .filter(Boolean);
  const label = names.length ? names.join(", ") : "Custom dish";
  return label.length > 120 ? `${label.slice(0, 117)}...` : label;
}

function renderEntryIngredients(entry) {
  const ingredients = Array.isArray(entry?.ingredients) ? entry.ingredients : [];
  if (!ingredients.length) return "";
  const items = ingredients
    .map((item) => {
      const line = formatIngredientLine(item);
      return line ? `<li>${escapeHtml(line)}</li>` : "";
    })
    .join("");
  return items ? `<ul class="entry-ingredients">${items}</ul>` : "";
}

function renderHistoryGroups() {
  if (!els.historyGroups) return;
  const entries = state.nutrition.history || [];
  if (!entries.length) {
    els.historyGroups.innerHTML = `<p class="empty-state">No dishes in this range.</p>`;
    return;
  }
  const groups = new Map();
  entries.forEach((entry) => {
    const dateKey = String(entry.consumed_at || "").slice(0, 10) || "unknown";
    const key = state.nutrition.historyView === "month" ? dateKey.slice(0, 7) : dateKey;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(entry);
  });
  els.historyGroups.innerHTML = Array.from(groups.entries())
    .map(([key, groupEntries]) => {
      const total = groupEntries.reduce((sum, entry) => sum + Number(entry.calories_kcal || 0), 0);
      const rows = groupEntries
        .map(
          (entry) => `
            <li class="history-entry">
              <div>
                <strong>${escapeHtml(entry.dish_name || "Dish")}</strong>
                <span>${escapeHtml(formatEntryTime(entry.consumed_at))} · ${escapeHtml(kcalText(entry.calories_kcal))}</span>
                ${renderEntryIngredients(entry)}
              </div>
              <div class="entry-actions">
                <button type="button" data-action="edit-history" data-id="${escapeHtml(entry.id)}">Edit</button>
                <button type="button" data-action="delete-history" data-id="${escapeHtml(entry.id)}">Delete</button>
              </div>
            </li>
          `,
        )
        .join("");
      return `
        <section class="history-group">
          <header>
            <h3>${escapeHtml(historyGroupLabel(key))}</h3>
            <span>${escapeHtml(kcalText(total))}</span>
          </header>
          <ul>${rows}</ul>
        </section>
      `;
    })
    .join("");
}

function renderNutritionState() {
  const profile = state.nutrition.profile || {};
  const requirement = state.nutrition.requirement || {};
  const summary = state.nutrition.summary || {};
  hydrateProfileForm(profile);
  renderBudgetCards(summary, requirement);
  renderRequirementDetails(profile, requirement);
  renderStats(summary);
  renderNutritionChart(summary);
  renderHistoryGroups();
}

function historyPayloadFromForm() {
  recalculateHistoryCaloriesFromIngredients();
  const payload = {
    dish_name: generatedDishName(),
    consumed_at: els.historyConsumedAt?.value || localDatetimeValue(),
    ingredients: state.nutrition.editIngredients.map((item) => recalculateIngredient(item)),
  };
  if (state.nutrition.pendingSource) {
    Object.assign(payload, state.nutrition.pendingSource);
  }
  return payload;
}

function resetHistoryForm() {
  if (!els.historyForm) return;
  els.historyEditId.value = "";
  els.historyConsumedAt.value = localDatetimeValue();
  state.nutrition.submitAttempted = false;
  state.nutrition.pendingSource = null;
  state.nutrition.editIngredients = [
    {
      name: "",
      count: null,
      portion_category: "normal",
      kcal: null,
    },
  ];
  renderIngredientEditor();
  if (els.historyComposerTitle) els.historyComposerTitle.textContent = "New consumed dish";
  els.historySubmitButton.textContent = "Add Dish";
  els.historyCancelEditButton.hidden = true;
}

function fillHistoryForm(entry) {
  if (!entry || !els.historyForm) return;
  els.historyEditId.value = entry.id || "";
  els.historyConsumedAt.value = String(entry.consumed_at || "").slice(0, 16);
  state.nutrition.submitAttempted = false;
  state.nutrition.pendingSource = null;
  state.nutrition.editIngredients = Array.isArray(entry.ingredients)
    ? entry.ingredients.map((item) => ingredientForEditor(item))
    : [];
  renderIngredientEditor();
  if (els.historyComposerTitle) els.historyComposerTitle.textContent = "Edit consumed dish";
  els.historySubmitButton.textContent = "Save Dish";
  els.historyCancelEditButton.hidden = false;
}

async function saveProfile(event) {
  event.preventDefault();
  try {
    const payload = await apiJson("/api/nutrition/profile", {
      method: "PUT",
      body: profilePayloadFromForm(),
    });
    state.nutrition.profile = payload.profile || {};
    state.nutrition.requirement = payload.requirement || {};
    showToast("Profile saved");
    await refreshNutrition();
  } catch (error) {
    showToast(error.message || "Profile save failed");
  }
}

async function saveHistoryEntry(event) {
  event.preventDefault();
  const validationMessage = validateHistoryIngredients();
  if (validationMessage) {
    showToast(validationMessage);
    return;
  }
  const entryId = els.historyEditId?.value || "";
  const method = entryId ? "PUT" : "POST";
  const path = entryId ? `/api/nutrition/history/${encodeURIComponent(entryId)}` : "/api/nutrition/history";
  try {
    await apiJson(path, { method, body: historyPayloadFromForm() });
    resetHistoryForm();
    showToast(entryId ? "Dish updated" : "Dish added");
    await refreshNutrition();
  } catch (error) {
    showToast(error.message || "History save failed");
  }
}

async function deleteHistoryEntry(entryId) {
  if (!entryId) return;
  if (!window.confirm("Delete this dish from diet history?")) return;
  try {
    await apiJson(`/api/nutrition/history/${encodeURIComponent(entryId)}`, { method: "DELETE" });
    showToast("Dish deleted");
    await refreshNutrition();
  } catch (error) {
    showToast(error.message || "Delete failed");
  }
}

function calorieDishName(result) {
  const ingredients = Array.isArray(result?.calories?.ingredients) ? result.calories.ingredients : [];
  const names = ingredients.map((item) => String(item.name || "").trim()).filter(Boolean);
  if (names.length) {
    const label = names.join(", ");
    return label.length > 120 ? `${label.slice(0, 117)}...` : label;
  }
  return "Estimated dish";
}

function calorieHistoryPayload(result) {
  const calories = result?.calories || {};
  const total = typeof calories.total_kcal === "number" ? calories.total_kcal : null;
  if (total === null) {
    throw new Error("No calorie total to save");
  }
  return {
    dish_name: calorieDishName(result),
    calories_kcal: Math.round(total * 10) / 10,
    consumed_at: localDatetimeValue(),
    source_job_id: state.lastJob?.id || "",
    source_run_dir: state.lastJob?.run_dir || "",
    image_name: state.imageName || "",
    estimation_scope: calories.estimation_scope || "",
    ingredients: Array.isArray(calories.ingredients) ? calories.ingredients : [],
  };
}

function calorieResultSource(calories) {
  return {
    source_job_id: state.lastJob?.id || "",
    source_run_dir: state.lastJob?.run_dir || "",
    image_name: state.imageName || "",
    estimation_scope: calories?.estimation_scope || "",
  };
}

async function addCalorieResultToHistory(button) {
  if (!state.lastResult || state.lastResult.task !== "calories") return;
  if (button) button.disabled = true;
  try {
    await apiJson("/api/nutrition/history", {
      method: "POST",
      body: calorieHistoryPayload(state.lastResult),
    });
    showToast("Dish added to diet history");
    await refreshNutrition();
  } catch (error) {
    showToast(error.message || "Could not add dish");
  } finally {
    if (button) button.disabled = false;
  }
}

async function reviewCalorieResultBeforeHistory(button) {
  if (!state.lastResult || state.lastResult.task !== "calories") return;
  const calories = state.lastResult.calories || {};
  const ingredients = Array.isArray(calories.ingredients) ? calories.ingredients : [];
  if (!ingredients.length) {
    showToast("No ingredient details to edit");
    return;
  }
  if (!els.historyForm) {
    showToast("Personal area is unavailable");
    return;
  }
  if (button) button.disabled = true;
  try {
    setActiveView("personal");
    await ensureIngredientOptionsLoaded();
    if (els.historyEditId) els.historyEditId.value = "";
    if (els.historyConsumedAt) els.historyConsumedAt.value = localDatetimeValue();
    state.nutrition.submitAttempted = false;
    state.nutrition.pendingSource = calorieResultSource(calories);
    state.nutrition.editIngredients = ingredients.map((item) => ingredientForEditor(item));
    renderIngredientEditor();
    if (els.historyComposerTitle) els.historyComposerTitle.textContent = "Review estimate before adding";
    if (els.historySubmitButton) els.historySubmitButton.textContent = "Add Dish";
    if (els.historyCancelEditButton) els.historyCancelEditButton.hidden = false;
    els.historyForm?.scrollIntoView({ behavior: "smooth", block: "start" });
    showToast("Review the ingredients, then add the dish");
  } catch (error) {
    showToast(error.message || "Could not open estimate editor");
  } finally {
    if (button) button.disabled = false;
  }
}

function renderInlineBudgetAfterCalories(total) {
  const summary = state.nutrition.summary || {};
  const today = summary.today || {};
  const limit = typeof today.limit_kcal === "number" ? today.limit_kcal : null;
  const consumed = typeof today.total_kcal === "number" ? today.total_kcal : 0;
  if (limit === null) {
    return `
      <div class="inline-budget no-limit">
        <span>Consumed today</span>
        <strong>${escapeHtml(kcalText(consumed))}</strong>
      </div>
    `;
  }
  const currentRemaining = limit - consumed;
  const projectedRemaining = typeof total === "number" ? currentRemaining - total : currentRemaining;
  const tone = projectedRemaining < 0 ? "over" : "under";
  return `
    <div class="inline-budget ${tone}">
      <span>Calories left today</span>
      <strong>${escapeHtml(kcalText(projectedRemaining))}</strong>
    </div>
  `;
}

function setHistoryView(view) {
  state.nutrition.historyView = view;
  els.historyDayViewButton?.classList.toggle("active", view === "day");
  els.historyMonthViewButton?.classList.toggle("active", view === "month");
  renderHistoryGroups();
}

function tooltipAttrs(text) {
  const value = escapeHtml(text);
  return `data-tooltip="${value}" aria-label="${value}"`;
}

function setAnswerVisible(visible) {
  els.answerText.hidden = !visible;
  els.outputZone.classList.toggle("answer-hidden", !visible);
}

function shouldHidePrimaryAnswer(result) {
  return result && (result.task === "task1" || result.task === "calories");
}

function setLatencyMetric(value) {
  els.latencyMetric.textContent = typeof value === "number" ? `Time ${value.toFixed(2)} s` : "Time --";
}

function normalizeRerankLabel(text, rerankApplied, lowerCase = false) {
  if (rerankApplied === true) return lowerCase ? "re-rank used" : "Re-rank used";
  if (rerankApplied === false) return lowerCase ? "re-rank skipped" : "Re-rank skipped";
  const value = String(text || "").toLowerCase();
  if (value.includes("used") || value.includes("applied")) {
    return lowerCase ? "re-rank used" : "Re-rank used";
  }
  if (value.includes("skip") || value.includes("siglip")) {
    return lowerCase ? "re-rank skipped" : "Re-rank skipped";
  }
  return "";
}

function task2ResultSection(result) {
  if (!result || typeof result !== "object") return null;
  if (result.task === "task2") return result;
  if (Array.isArray(result.sections)) {
    return result.sections.find((section) => section && section.task === "task2") || null;
  }
  return null;
}

function updateTaskButtonRuntime(result) {
  const task2 = task2ResultSection(result);
  if (!task2) return;
  const button = els.taskButtons.find((item) => item.dataset.task === "task2");
  const subtitle = button ? button.querySelector(".task-subtitle") : null;
  if (!subtitle) return;
  subtitle.textContent =
    normalizeRerankLabel(task2.rerank_or_vlm, task2.rerank_applied, true) || TASK_DEFAULT_SUBTITLES.task2;
}

function usageLabelForResult(result) {
  if (!result || !result.rerank_or_vlm) return "";
  if (result.task === "task2") {
    return normalizeRerankLabel(result.rerank_or_vlm, result.rerank_applied, false) || result.rerank_or_vlm;
  }
  return result.rerank_or_vlm;
}

function usagePillColor(result) {
  if (result && result.task === "calories") return "blue";
  return "coral";
}

function taskPillColor(result) {
  if (result && result.task === "calories") return "coral";
  return "blue";
}

function renderResult(result) {
  state.lastResult = result || null;
  els.diagnosticsButton.disabled = !state.lastResult;
  const answer = result.answer || result.spoken_answer || "No output returned.";
  const hideAnswer = shouldHidePrimaryAnswer(result);
  setAnswerVisible(!hideAnswer);
  els.answerText.textContent = hideAnswer ? "" : answer;
  setLatencyMetric(result.latency_sec);
  const meta = [];
  if (result.task) meta.push(pill(TASK_LABELS[result.task] || result.task, taskPillColor(result)));
  const usageLabel = usageLabelForResult(result);
  if (usageLabel) meta.push(pill(usageLabel, usagePillColor(result)));
  els.outputMeta.innerHTML = meta.join("");
  els.resultDetails.innerHTML = renderDetails(result);
  updateTaskButtonRuntime(result);
  speakResult(result.spoken_answer || answer);
}

function diagnosticsSections(result) {
  if (!result || typeof result !== "object") return [];
  const sections = [];
  if (hasDiagnosticsData(result)) {
    sections.push(result);
  }
  if (Array.isArray(result.sections) && result.sections.length) {
    sections.push(...result.sections.filter((section) => section && typeof section === "object"));
  }
  if (sections.length) {
    return sections;
  }
  return [result];
}

function diagnosticsOf(section) {
  return section && typeof section.diagnostics === "object" ? section.diagnostics : {};
}

function hasDiagnosticsData(section) {
  const diagnostics = diagnosticsOf(section);
  const timings = Array.isArray(diagnostics.timings) ? diagnostics.timings : [];
  const power = diagnostics.power && typeof diagnostics.power === "object" ? diagnostics.power : {};
  const resources = diagnostics.resources && typeof diagnostics.resources === "object" ? diagnostics.resources : {};
  return timings.length > 0 || Object.keys(power).length > 0 || Object.keys(resources).length > 0;
}

function statFromResources(resources, key, field = "avg") {
  const stats = resources?.stats?.[key];
  const value = stats && typeof stats[field] === "number" ? stats[field] : null;
  return value;
}

function maxTimingValue(rows) {
  const values = rows.map((row) => (typeof row.sec === "number" ? row.sec : 0));
  return Math.max(0.001, ...values);
}

function renderMetricCard(label, value, sub = "") {
  return `
    <div class="diagnostic-card">
      <span class="diagnostic-label">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${sub ? `<span class="diagnostic-sub">${escapeHtml(sub)}</span>` : ""}
    </div>
  `;
}

function renderBarRow(label, valueText, ratio, tone = "") {
  const pct = Math.max(2, Math.min(100, Number.isFinite(ratio) ? ratio * 100 : 0));
  return `
    <li class="diagnostic-bar-row ${tone}">
      <div class="diagnostic-bar-label">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(valueText)}</strong>
      </div>
      <div class="diagnostic-bar-track"><span style="width:${pct.toFixed(1)}%"></span></div>
    </li>
  `;
}

function renderTimingPanel(section) {
  const diagnostics = diagnosticsOf(section);
  const rows = Array.isArray(diagnostics.timings) ? diagnostics.timings : [];
  if (!rows.length) {
    return `<section class="diagnostic-panel"><h3>Timing</h3><p class="diagnostic-empty">No timing breakdown returned.</p></section>`;
  }
  const maxValue = maxTimingValue(rows);
  const html = rows
    .map((row) =>
      renderBarRow(row.label || row.key || "stage", `${formatMetric(row.sec, " s", row.sec < 1 ? 3 : 2)}`, row.sec / maxValue),
    )
    .join("");
  return `<section class="diagnostic-panel"><h3>Timing breakdown</h3><ul class="diagnostic-bars">${html}</ul></section>`;
}

function renderPowerPanel(section) {
  const diagnostics = diagnosticsOf(section);
  const power = diagnostics.power || {};
  const resources = diagnostics.resources || {};
  if (!Object.keys(power).length && !Object.keys(resources).length) {
    return "";
  }
  const enabled = power.enabled !== false;
  const available = power.available !== false;
  const sampleCount = typeof power.sample_count === "number" ? power.sample_count : 0;
  const cards = [
    renderMetricCard("Avg power", formatMetric(power.avg_power_w, " W", 2), power.selected_power_source || ""),
    renderMetricCard("Max power", formatMetric(power.max_power_w, " W", 2)),
    renderMetricCard("Energy", formatMetric(power.energy_j, " J", 2), "for this image run"),
    renderMetricCard("Samples", sampleCount ? String(sampleCount) : "--", enabled && available ? "tegrastats" : "not sampled"),
  ].join("");

  const resourceRows = [];
  const cpuP95 = statFromResources(resources, "cpu_all_core_avg_util_pct", "p95");
  const gpuP95 = statFromResources(resources, "gpu_gr3d_util_pct", "p95");
  const ramAvg = statFromResources(resources, "ram_used_mb");
  const ramTotal = statFromResources(resources, "ram_total_mb", "max");
  if (cpuP95 !== null) resourceRows.push(renderBarRow("CPU p95", `${formatNumber(cpuP95, 1)}%`, cpuP95 / 100, "green"));
  if (gpuP95 !== null) resourceRows.push(renderBarRow("GPU p95", `${formatNumber(gpuP95, 1)}%`, gpuP95 / 100, "blue"));
  if (ramAvg !== null) {
    const denom = ramTotal || ramAvg;
    resourceRows.push(
      renderBarRow(
        "RAM avg",
        `${formatNumber(ramAvg / 1024, 1)} GB${ramTotal ? ` / ${formatNumber(ramTotal / 1024, 1)} GB` : ""}`,
        ramAvg / denom,
        "coral",
      ),
    );
  }

  const railRows = Object.entries(power.rail_avg_power_w || {})
    .filter(([rail]) => ["VDD_GPU_SOC", "VDD_CPU_CV"].includes(String(rail).toUpperCase()))
    .filter(([, value]) => typeof value === "number")
    .map(([rail, value]) => renderBarRow(rail, `${formatMetric(value, " W", 2)}`, value / Math.max(1, power.max_power_w || value), "yellow"))
    .join("");

  const unavailable = !enabled
    ? "Power diagnostics are disabled for this server run."
    : !available
      ? "tegrastats was not available on this machine."
      : sampleCount
        ? ""
        : power.error || "No resource samples were captured for this short run.";

  return `
    <section class="diagnostic-panel">
      <h3>Power and resources</h3>
      <div class="diagnostic-card-grid">${cards}</div>
      ${unavailable ? `<p class="diagnostic-empty">${escapeHtml(unavailable)}</p>` : ""}
      ${resourceRows.length ? `<ul class="diagnostic-bars">${resourceRows.join("")}</ul>` : ""}
      ${railRows ? `<h4>Power rails</h4><ul class="diagnostic-bars compact">${railRows}</ul>` : ""}
    </section>
  `;
}

function renderSystemPanel(section) {
  const system = diagnosticsOf(section).system || {};
  if (!Object.keys(system).length) {
    return "";
  }
  return `
    <section class="diagnostic-panel">
      <h3>System</h3>
      <div class="system-grid">
        <span>Device</span><strong>${escapeHtml(system.device_model || "Jetson Orin")}</strong>
        <span>RAM</span><strong>${escapeHtml(system.ram_total_label || "64 GB")}</strong>
        <span>Platform</span><strong>${escapeHtml(system.platform || "--")}</strong>
      </div>
    </section>
  `;
}

function renderDiagnosticsSection(section) {
  const title = TASK_LABELS[section.task] || section.task || "Run";
  const latency = typeof section.latency_sec === "number" ? `${section.latency_sec.toFixed(2)} s` : "--";
  return `
    <article class="diagnostic-task">
      <div class="diagnostic-task-header">
        <h3>${escapeHtml(title)}</h3>
        <span>${escapeHtml(latency)}</span>
      </div>
      ${renderTimingPanel(section)}
      ${renderPowerPanel(section)}
      ${renderSystemPanel(section)}
    </article>
  `;
}

function openDiagnosticsModal() {
  if (!state.lastResult) return;
  const sections = diagnosticsSections(state.lastResult);
  els.diagnosticsContent.innerHTML = sections.map(renderDiagnosticsSection).join("");
  els.diagnosticsModal.hidden = false;
}

function closeDiagnosticsModal() {
  els.diagnosticsModal.hidden = true;
}

function renderDetails(result) {
  if (!result || typeof result !== "object") return "";
  if (Array.isArray(result.sections) && result.sections.length) {
    return result.sections
      .map(
        (section) => `
          <section class="detail-section">
            <p class="detail-title">${escapeHtml(TASK_LABELS[section.task] || section.task || "Result")}</p>
            ${
              shouldHidePrimaryAnswer(section)
                ? ""
                : `<div class="answer-text small">${escapeHtml(section.answer || "")}</div>`
            }
            ${renderDetails(section)}
          </section>
        `,
      )
      .join("");
  }
  if (result.task === "task1") {
    const labels = Array.isArray(result.selected_labels) ? result.selected_labels : [];
    if (!labels.length) return "";
    return `
      <section class="detail-section primary-detail">
        <p class="detail-title">Selected labels</p>
        <div class="label-chip-grid">${labels.map((label) => `<span class="label-chip">${escapeHtml(label)}</span>`).join("")}</div>
      </section>
    `;
  }
  if (result.task === "task2") {
    const candidates = Array.isArray(result.top_candidates) ? result.top_candidates : [];
    const scores = formatCandidateScores(candidates.map((item) => item.score));
    const rows = candidates
      .map((item, index) => {
        const score = scores[index];
        return `
          <li class="detail-row">
            <span class="detail-main">
              ${index + 1}. ${escapeHtml(item.caption || "")}
            </span>
            <span
              class="detail-value metric-chip score-chip has-tooltip"
              ${tooltipAttrs("Confidence score used to rank caption candidates for this image. Higher is better within this image. Extra decimals appear only when close scores need to be distinguished.")}
            >${escapeHtml(score)}</span>
          </li>
        `;
      })
      .join("");
    return `
      <section class="detail-section">
        <p class="detail-title">Caption candidates</p>
        <ul class="detail-list">${rows || "<li class='detail-row'><span class='detail-main'>No candidates returned</span></li>"}</ul>
      </section>
    `;
  }
  if (result.task === "calories") {
    return renderCalories(result.calories || {});
  }
  return "";
}

function renderCalories(calories) {
  const total = calories.total_kcal;
  const perInstanceScope = calories.estimation_scope === "per_instance_not_full_image";
  const averageScope = perInstanceScope || calories.estimation_scope === "average_portion_not_full_image";
  const ingredients = Array.isArray(calories.ingredients) ? calories.ingredients : [];
  const rows = ingredients
    .map((item) => {
      const metrics = calorieMetricChips(item, calories.estimation_scope);
      return `
        <li class="detail-row calorie-row">
          <span class="detail-main calorie-name">${escapeHtml(item.name || "ingredient")}</span>
          <span class="calorie-metrics">${metrics}</span>
        </li>
      `;
    })
    .join("");
  const scopeLabel = perInstanceScope
    ? "Per representative item"
    : averageScope
      ? "Average eating portion"
      : "Full dish estimate";
  return `
    <section class="detail-section primary-detail calorie-detail">
      <p class="detail-title">${perInstanceScope ? "Per-item estimate" : averageScope ? "Average portion estimate" : "Calorie estimate"}</p>
      <div class="calorie-summary">
        ${typeof total === "number" ? `<span class="calorie-total">${Math.round(total)} kcal</span>` : ""}
        <span class="calorie-scope">${escapeHtml(scopeLabel)}</span>
      </div>
      <ul class="detail-list">${rows}</ul>
      ${renderInlineBudgetAfterCalories(total)}
      <div class="result-action-row">
        <button class="primary-action" type="button" data-action="add-calorie-result" ${typeof total === "number" ? "" : "disabled"}>
          Add to diet history
        </button>
        <button class="secondary-action" type="button" data-action="review-calorie-result" ${ingredients.length ? "" : "disabled"}>
          Edit before adding
        </button>
      </div>
    </section>
  `;
}

function metricChip(text, tooltip, tone = "") {
  if (!text) return "";
  const classes = ["metric-chip"];
  if (tone) classes.push(tone);
  if (tooltip) classes.push("has-tooltip");
  return `<span class="${classes.join(" ")}" ${tooltip ? tooltipAttrs(tooltip) : ""}>${escapeHtml(text)}</span>`;
}

function calorieMetricChips(item, estimationScope) {
  const chips = [];
  if (item.abundance_scene) {
    chips.push(
      metricChip(
        "count: many visible",
        "The image looks like a display or abundance scene. The app reports calories per representative item instead of summing every visible item.",
        "green",
      ),
    );
  } else if (typeof item.count === "number") {
    chips.push(
      metricChip(
        `count: ${formatCount(item.count)}`,
        "Count is the model estimate of visible ingredient instances. Fractional counts are rounded to the nearest half step for display.",
        "green",
      ),
    );
  }

  const portion = String(item.portion_category || "").trim();
  if (!item.abundance_scene && portion && portion !== "none") {
    chips.push(
      metricChip(
        `portion: ${portion}`,
        "Portion is the model size category for this ingredient. It is used to scale the calorie estimate.",
        "green",
      ),
    );
  } else if (item.abundance_scene || estimationScope === "per_instance_not_full_image") {
    chips.push(
      metricChip(
        "portion: one item",
        "For abundance scenes, this is a per-item estimate rather than a full-image total.",
        "green",
      ),
    );
  }

  if (typeof item.per_instance_kcal === "number" && item.abundance_scene) {
    chips.push(metricChip(`${Math.round(item.per_instance_kcal)} kcal each`, "", "coral"));
  } else if (typeof item.kcal === "number") {
    chips.push(metricChip(`${Math.round(item.kcal)} kcal`, "", "coral"));
  }
  return chips.join("");
}

function formatCount(value) {
  if (!Number.isFinite(value)) return "";
  const rounded = Math.max(0.5, Math.floor(value * 2 + 0.5) / 2);
  if (Math.abs(rounded - Math.round(rounded)) < 1e-6) return String(Math.round(rounded));
  return rounded.toFixed(1).replace(/0+$/, "").replace(/\.$/, "");
}

function numericScore(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function truncatedScoreKey(value, decimals) {
  const scale = 10 ** decimals;
  const truncated = Math.trunc(value * scale) / scale;
  return truncated.toFixed(decimals);
}

function expandSimilarScoreGroups(numericValues, decimals, places, keyFn) {
  const groups = new Map();
  numericValues.forEach((value, index) => {
    if (value === null) return;
    const key = keyFn(value);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(index);
  });

  let expanded = false;
  groups.forEach((indices) => {
    if (indices.length < 2) return;
    const values = indices.map((index) => numericValues[index]);
    const spread = Math.max(...values) - Math.min(...values);
    if (spread < 1e-12) return;
    indices.forEach((index) => {
      if (decimals[index] <= places) {
        decimals[index] = places + 1;
        expanded = true;
      }
    });
  });
  return expanded;
}

function formatCandidateScores(values) {
  const numericValues = values.map(numericScore);
  const decimals = numericValues.map((value) => (value === null ? 0 : 1));

  for (let places = 1; places < 3; places += 1) {
    const roundedExpanded = expandSimilarScoreGroups(
      numericValues,
      decimals,
      places,
      (value) => value.toFixed(places),
    );
    const truncatedExpanded = expandSimilarScoreGroups(
      numericValues,
      decimals,
      places,
      (value) => truncatedScoreKey(value, places),
    );
    const expanded = roundedExpanded || truncatedExpanded;
    if (!expanded) break;
  }

  return values.map((value, index) => formatScore(value, decimals[index]));
}

function formatScore(value, decimals = 1) {
  const numericValue = numericScore(value);
  if (numericValue !== null) return numericValue.toFixed(decimals);
  if (value === null || value === undefined || value === "") {
    return "";
  }
  return String(value);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Status unavailable");
    if (typeof payload.sample_images === "number") {
      state.sampleCount = payload.sample_images;
    }
    renderPreloadStatus(payload.preload || { status: payload.mock_backend ? "skipped" : "idle", events: [] });
  } catch (error) {
    console.warn(error);
  }
}

function bindEvents() {
  els.pipelineViewButton?.addEventListener("click", () => setActiveView("pipeline"));
  els.personalViewButton?.addEventListener("click", () => setActiveView("personal"));
  els.profileForm?.addEventListener("submit", saveProfile);
  els.historyRefreshButton?.addEventListener("click", () => refreshNutrition());
  els.historyDayViewButton?.addEventListener("click", () => setHistoryView("day"));
  els.historyMonthViewButton?.addEventListener("click", () => setHistoryView("month"));
  els.historyForm?.addEventListener("submit", saveHistoryEntry);
  els.historyCancelEditButton?.addEventListener("click", resetHistoryForm);
  els.addIngredientButton?.addEventListener("click", addManualIngredient);
  els.historyIngredients?.addEventListener("change", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const row = target?.closest(".ingredient-edit-row");
    if (!row) return;
    if (target?.matches('[data-ingredient-control="remove"]')) {
      return;
    }
    syncIngredientFromRow(row);
    if (target?.matches('[data-ingredient-control="mode"], [data-ingredient-control="name"]')) {
      renderIngredientEditor();
      return;
    }
    recalculateHistoryCaloriesFromIngredients();
  });
  els.historyIngredients?.addEventListener("input", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target?.matches('[data-ingredient-control="count"], [data-ingredient-control="name"]')) return;
    if (target.matches('[data-ingredient-control="name"]') && ingredientReference(target.value)) {
      const row = target.closest(".ingredient-edit-row");
      if (row) syncIngredientFromRow(row);
      renderIngredientEditor();
      return;
    }
    recalculateHistoryCaloriesFromIngredients();
  });
  els.historyIngredients?.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target?.matches('[data-ingredient-control="remove"]')) return;
    const row = target.closest(".ingredient-edit-row");
    const index = Number(row?.dataset?.ingredientIndex);
    removeIngredientAt(index);
  });
  els.historyGroups?.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest("button[data-action]");
    if (!button) return;
    const entryId = button.dataset.id || "";
    if (button.dataset.action === "edit-history") {
      const entry = state.nutrition.history.find((item) => item.id === entryId);
      fillHistoryForm(entry);
    }
    if (button.dataset.action === "delete-history") {
      deleteHistoryEntry(entryId);
    }
  });
  els.resultDetails?.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest("button[data-action]");
    if (!button) return;
    if (button.dataset.action === "add-calorie-result") {
      addCalorieResultToHistory(button);
    }
    if (button.dataset.action === "review-calorie-result") {
      reviewCalorieResultBeforeHistory(button);
    }
  });
  els.taskButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const task = button.dataset.task || "task1";
      setSelectedTask(task);
      startJob(task);
    });
  });
  els.cameraButton.addEventListener("click", startCamera);
  els.uploadButton.addEventListener("click", () => els.imageUpload.click());
  els.prevImageButton.addEventListener("click", () => {
    const count = state.sampleCount || 1;
    loadDatasetImage((state.sampleIndex - 1 + count) % count);
  });
  els.nextImageButton.addEventListener("click", () => {
    const count = state.sampleCount || 1;
    loadDatasetImage((state.sampleIndex + 1) % count);
  });
  els.imageUpload.addEventListener("change", (event) => {
    handleUpload(event.target.files?.[0]);
    event.target.value = "";
  });
  els.soundButton.addEventListener("click", () => {
    const enabled = !state.soundEnabled;
    setSoundState(enabled);
    if (enabled) {
      speakResult("Sound on.");
    }
  });
  els.soundButton.addEventListener("mouseenter", () => showFloatingTooltip(els.soundButton));
  els.soundButton.addEventListener("mouseleave", hideFloatingTooltip);
  els.soundButton.addEventListener("focus", () => showFloatingTooltip(els.soundButton));
  els.soundButton.addEventListener("blur", hideFloatingTooltip);
  els.diagnosticsButton.addEventListener("click", openDiagnosticsModal);
  els.diagnosticsButton.addEventListener("mouseenter", () => showFloatingTooltip(els.diagnosticsButton));
  els.diagnosticsButton.addEventListener("mouseleave", hideFloatingTooltip);
  els.diagnosticsButton.addEventListener("focus", () => showFloatingTooltip(els.diagnosticsButton));
  els.diagnosticsButton.addEventListener("blur", hideFloatingTooltip);
  els.diagnosticsCloseButton.addEventListener("click", closeDiagnosticsModal);
  els.diagnosticsModal.addEventListener("click", (event) => {
    if (event.target === els.diagnosticsModal) {
      closeDiagnosticsModal();
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !els.diagnosticsModal.hidden) {
      closeDiagnosticsModal();
    }
  });
  window.addEventListener("resize", positionFloatingTooltip);
  window.addEventListener("scroll", positionFloatingTooltip, true);
  els.voiceButton.addEventListener("click", () => {
    startVoiceCapture();
  });
}

function init() {
  bindEvents();
  setSoundState(true);
  setSelectedTask("task1");
  setDefaultNutritionRange();
  resetHistoryForm();
  renderPreloadStatus({ status: "idle", events: [] });
  refreshStatus();
  refreshNutrition();
  window.setInterval(refreshStatus, 1200);
}

init();
