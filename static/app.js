import { LabelSmoother, RepCounter, instantGeometry } from "./temporal.mjs";
const $ = (id) => document.getElementById(id);
const names = {
  standing: "Standing",
  sitting: "Sitting",
  marching: "Marching in place",
  squats: "Squats",
  jumping_jacks: "Jumping jacks",
  waving: "Waving",
  unknown: "Uncertain",
};
const shortNames = {
  standing: "Stand",
  sitting: "Sit",
  marching: "March",
  squats: "Squat",
  jumping_jacks: "Jack",
  waving: "Wave",
  unknown: "Unknown",
};
const links = [
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
let mode = null,
  stream = null,
  objectUrl = null,
  detector = null,
  frames = [],
  samples = [],
  recording = null,
  sessionLog = [];
let timer = null,
  startTime = 0,
  lastSample = -1,
  lastPrediction = -1,
  busy = false,
  generation = 0,
  aspect = 1,
  lastVideoTime = -1;
let demoFrames = null,
  demoLabel = null,
  loading = false,
  recordPreparing = false;
let smoother = new LabelSmoother(),
  counter = new RepCounter();
const ctx = $("canvas").getContext("2d");
const noPose = () => Array.from({ length: 33 }, () => [0, 0, 0, 0]);
for (const id of ["demo-label", "record-label"])
  for (const [key, title] of Object.entries(names)) {
    if (key === "unknown") continue;
    const o = document.createElement("option");
    o.value = key;
    o.textContent = title;
    $(id).append(o);
  }

function notice(message) {
  $("notice").textContent = message;
  $("notice").classList.toggle("hidden", !message);
}
async function api(path, body) {
  const r = await fetch(
    path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  let data;
  try {
    data = await r.json();
  } catch {
    throw new Error("The local app returned an unreadable response.");
  }
  if (!r.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
}
function safe(action) {
  return async (...args) => {
    try {
      await action(...args);
    } catch (e) {
      notice(e.message || String(e));
    }
  };
}
function el(tag, text, className) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (className) e.className = className;
  return e;
}
function tableRow(values, header = false) {
  const row = el("tr");
  for (const value of values)
    row.append(el(header ? "th" : "td", String(value)));
  return row;
}
function updateCounts() {
  $("squat-count").textContent = counter.counts.squats;
  $("jack-count").textContent = counter.counts.jumping_jacks;
}
function clearPrediction(reason = "Start a source to see a prediction.") {
  $("activity").textContent = "Waiting";
  $("reason").textContent = reason;
  $("quality").textContent = "—";
  $("quality-bar").style.width = "0";
  $("confidence").textContent = "—";
  $("left-knee").textContent = "—";
  $("right-knee").textContent = "—";
}
function controls() {
  const running = !!mode;
  for (const id of ["camera", "demo", "video-file"]) $(id).disabled = loading;
  $("stop").disabled = !running;
  $("record").disabled =
    !running || mode === "synthetic" || !!recording || recordPreparing;
  $("stream-dot").classList.toggle("live", running);
}
function stopSource() {
  generation++;
  mode = null;
  clearTimeout(timer);
  timer = null;
  frames = [];
  samples = [];
  smoother.reset();
  counter.interrupt();
  lastSample = -1;
  lastPrediction = -1;
  lastVideoTime = -1;
  busy = false;
  if (recording || recordPreparing) {
    recording = null;
    recordPreparing = false;
    $("record-state").textContent =
      "Recording cancelled when the input changed.";
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  $("video").pause();
  $("video").srcObject = null;
  $("video").removeAttribute("src");
  $("video").load();
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  $("source-label").textContent = "Ready to begin";
  $("sample-rate").textContent = "10 Hz target";
  $("latency").textContent = "PoseLab / Local";
  $("source-badge").classList.add("hidden");
  $("stage-empty").classList.remove("hidden");
  ctx.clearRect(0, 0, $("canvas").width, $("canvas").height);
  clearPrediction();
  controls();
}
async function ensureDetector() {
  if (detector) return;
  const status = await api("/api/status");
  if (!status.pose_assets_ready)
    throw new Error(
      "Pose files are missing. Run python scripts/setup_assets.py, then try again.",
    );
  $("source-label").textContent = "Loading pose detector…";
  const { FilesetResolver, PoseLandmarker } = await import(
    "./vendor/vision_bundle.mjs"
  );
  const vision = await FilesetResolver.forVisionTasks("/static/vendor/wasm");
  detector = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/static/models/pose_landmarker_lite.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}
function beginSource(source) {
  mode = source;
  startTime = performance.now();
  frames = [];
  samples = [];
  lastSample = -1;
  lastPrediction = -1;
  sessionLog = [];
  counter.reset();
  updateCounts();
  smoother.reset();
  $("stage-empty").classList.add("hidden");
  $("source-badge").textContent =
    source === "synthetic"
      ? "SIMULATED LANDMARKS · NOT RESEARCH DATA"
      : source === "camera"
        ? "CAMERA · LOCAL PROCESSING"
        : "LOCAL VIDEO · NO UPLOAD";
  $("source-badge").classList.remove("hidden");
  $("source-label").textContent =
    source === "synthetic"
      ? `Simulation / ${names[demoLabel]}`
      : source === "camera"
        ? "Camera connected"
        : "Video playback";
  clearPrediction("Collecting a two-second movement window.");
  controls();
  tick();
}
async function camera() {
  loading = true;
  stopSource();
  controls();
  notice("");
  try {
    await ensureDetector();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    $("video").srcObject = stream;
    await $("video").play();
    aspect = $("video").videoWidth / $("video").videoHeight;
    beginSource("camera");
  } catch (e) {
    stopSource();
    if (["NotAllowedError", "PermissionDeniedError"].includes(e.name))
      throw new Error(
        "Camera permission was declined. Allow it in the browser, or use a local video.",
      );
    throw e;
  } finally {
    loading = false;
    controls();
  }
}
async function openVideo(event) {
  const file = event.target.files[0];
  if (!file) return;
  loading = true;
  stopSource();
  controls();
  notice("");
  try {
    await ensureDetector();
    objectUrl = URL.createObjectURL(file);
    $("video").src = objectUrl;
    await $("video").play();
    aspect = $("video").videoWidth / $("video").videoHeight;
    beginSource("video");
  } catch (e) {
    stopSource();
    throw new Error(
      `Could not play this video. Try an MP4 supported by your browser. ${e.message}`,
    );
  } finally {
    loading = false;
    controls();
    event.target.value = "";
  }
}
async function simulation() {
  loading = true;
  stopSource();
  controls();
  notice("");
  try {
    demoLabel = $("demo-label").value;
    const demo = await api(`/api/demo/${demoLabel}`);
    demoFrames = demo.frames;
    aspect = 1;
    $("engine").value = "rules";
    beginSource("synthetic");
  } finally {
    loading = false;
    controls();
  }
}

function draw(p) {
  const canvas = $("canvas");
  const w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!p) return;
  const dw = aspect > w / h ? w : h * aspect,
    dh = dw / aspect,
    ox = (w - dw) / 2,
    oy = (h - dh) / 2;
  const xy = (i) => [ox + p[i][0] * dw, oy + p[i][1] * dh];
  if (mode === "synthetic") {
    ctx.fillStyle = "#95a99812";
    ctx.beginPath();
    ctx.ellipse(w / 2, h * 0.94, 100, 12, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  for (const [a, b] of links) {
    if (p[a][3] < 0.5 || p[b][3] < 0.5) continue;
    ctx.strokeStyle = a % 2 ? "#d6e998" : "#80bac1";
    ctx.beginPath();
    ctx.moveTo(...xy(a));
    ctx.lineTo(...xy(b));
    ctx.stroke();
  }
  for (const i of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    if (p[i][3] < 0.5) continue;
    ctx.fillStyle = "#edf5d7";
    ctx.beginPath();
    ctx.arc(...xy(i), 5, 0, Math.PI * 2);
    ctx.fill();
  }
  if (mode === "synthetic") {
    ctx.strokeStyle = "#d6e998";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(...xy(0), 22, 0, Math.PI * 2);
    ctx.stroke();
  }
}
async function predict(myGeneration) {
  busy = true;
  lastPrediction = frames.at(-1).t;
  const source = mode,
    engine = $("engine").value;
  const started = performance.now();
  try {
    const r = await api("/api/predict", {
      frames: frames.slice(),
      aspect,
      engine,
    });
    if (myGeneration !== generation) return;
    const label = smoother.update(r.label);
    $("activity").textContent =
      label === "unknown"
        ? r.label === "unknown"
          ? "Uncertain"
          : "Confirming…"
        : names[label];
    $("reason").textContent = r.reason;
    $("quality").textContent = `${Math.round(r.quality * 100)}%`;
    $("quality-bar").style.width = `${r.quality * 100}%`;
    $("confidence").textContent =
      r.score === null ? "—" : `${Math.round(r.score * 100)}%`;
    $("latency").textContent =
      `Prediction request ${Math.round(performance.now() - started)} ms`;
    sessionLog.push({
      seconds: Number(lastPrediction.toFixed(2)),
      source,
      engine,
      raw_activity: r.label,
      display_activity: label,
      visibility: r.quality,
      vote_score: r.score ?? "",
      squats: counter.counts.squats,
      jumping_jacks: counter.counts.jumping_jacks,
    });
    if (sessionLog.length > 7200) sessionLog.shift();
  } catch (e) {
    if (myGeneration === generation) {
      smoother.reset();
      counter.interrupt();
      clearPrediction("Prediction unavailable.");
      notice(e.message);
    }
  } finally {
    if (myGeneration === generation) busy = false;
  }
}
function tick() {
  if (!mode) return;
  const tickStarted = performance.now();
  const myGeneration = generation;
  let t, p;
  try {
    if (mode === "synthetic") {
      t = (performance.now() - startTime) / 1000;
      const index = Math.floor(t * 10) % 80;
      p = demoFrames[index].landmarks;
    } else {
      const video = $("video");
      if (video.ended) {
        if (recording)
          notice(
            "Video ended before the clip was complete. Use a longer segment.",
          );
        stopSource();
        return;
      }
      if (video.readyState < 2 || video.currentTime === lastVideoTime) {
        timer = setTimeout(tick, 40);
        return;
      }
      lastVideoTime = video.currentTime;
      t =
        mode === "video"
          ? video.currentTime
          : (performance.now() - startTime) / 1000;
      const pose = detector.detectForVideo(video, performance.now());
      p =
        pose.landmarks?.[0]?.map((v) => [v.x, v.y, v.z, v.visibility ?? 0]) ??
        noPose();
    }
    if (t - lastSample < 0.085) {
      timer = setTimeout(tick, 20);
      return;
    }
    lastSample = t;
    const frame = { t, landmarks: p };
    frames.push(frame);
    frames = frames.filter((f) => t - f.t <= 2.08);
    samples.push(performance.now());
    samples = samples.filter((s) => performance.now() - s <= 2000);
    $("sample-rate").textContent =
      `${(samples.length / 2).toFixed(1)} pose samples/s`;
    draw(p);
    const geometry = instantGeometry(p, aspect);
    if (geometry) {
      $("left-knee").textContent = `${Math.round(geometry.left_knee * 180)}°`;
      $("right-knee").textContent = `${Math.round(geometry.right_knee * 180)}°`;
    } else {
      smoother.reset();
      counter.interrupt();
      clearPrediction("Keep the full body visible.");
    }
    counter.update(smoother.label, geometry, t);
    updateCounts();
    if (recording) {
      recording.frames.push(frame);
      const elapsed = t - recording.start;
      $("record-state").textContent =
        `Recording ${names[recording.label]} · ${Math.min(elapsed, 8).toFixed(1)} / 8.0 seconds`;
      if (elapsed >= 8) finishRecording();
    }
    if (!busy && t - lastPrediction >= 0.48 && frames.length >= 12)
      predict(myGeneration);
  } catch (e) {
    stopSource();
    notice(`Pose processing stopped: ${e.message}`);
    return;
  }
  if (myGeneration === generation)
    timer = setTimeout(
      tick,
      Math.max(5, 100 - (performance.now() - tickStarted)),
    );
}
async function record() {
  if (!mode || mode === "synthetic" || recording || recordPreparing) return;
  if (!$("consent").checked)
    throw new Error("Confirm participant permission before saving landmarks.");
  const subject = $("subject").value.trim();
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(subject))
    throw new Error(
      "Use a participant ID with 1–32 letters, digits, hyphens or underscores.",
    );
  const label = $("record-label").value;
  const source = mode;
  const myGeneration = generation;
  recordPreparing = true;
  controls();
  for (let n = 3; n > 0; n--) {
    $("record-state").textContent =
      `Begin ${names[label]} · recording starts in ${n}…`;
    await new Promise((r) => setTimeout(r, 1000));
    if (generation !== myGeneration) return;
  }
  recordPreparing = false;
  recording = {
    frames: [],
    subject,
    label,
    source,
    consent: true,
    aspect,
    start: frames.at(-1)?.t ?? 0,
  };
  controls();
}
async function finishRecording() {
  const clip = recording;
  recording = null;
  controls();
  const { start, ...body } = clip;
  $("record-state").textContent = "Saving landmark clip…";
  try {
    const saved = await api("/api/clips", body);
    $("record-state").textContent =
      `Saved ${names[saved.label]} for ${saved.subject} · ${saved.frames} frames`;
    await refreshDataset();
  } catch (e) {
    $("record-state").textContent = "Clip was not saved.";
    notice(e.message);
  }
}
async function refreshStatus() {
  const s = await api("/api/status");
  const option = $("engine").querySelector("[value=trained]");
  option.disabled = !s.trained;
  option.textContent = s.trained
    ? "Trained temporal classifier"
    : "Trained classifier · collect data first";
}
async function refreshDataset() {
  const data = await api("/api/dataset");
  $("clip-total").textContent = `${data.total_clips} clips`;
  $("dataset-empty").classList.toggle("hidden", data.total_clips > 0);
  $("dataset-table")
    .querySelector("thead")
    .replaceChildren(
      tableRow(
        ["Participant", ...data.labels.map((l) => shortNames[l]), "Total"],
        true,
      ),
    );
  const body = $("dataset-table").querySelector("tbody");
  body.replaceChildren();
  for (const s of data.subjects) {
    const counts = data.labels.map((l) => data.counts[s][l]);
    body.append(tableRow([s, ...counts, counts.reduce((a, b) => a + b, 0)]));
  }
}
const percent = (x) => (x === null ? "—" : `${(x * 100).toFixed(1)}%`);
function renderResults(r) {
  $("results-empty").classList.toggle("hidden", !r.pending);
  $("results-content").classList.toggle("hidden", !!r.pending);
  $("download-results").classList.toggle("hidden", !!r.pending);
  if (r.pending) return;
  $("train-state").textContent = "A trained model is available in Live lab.";
  $("run-info").textContent =
    `Run ${r.run_id} · Test participants: ${r.split_subjects.test.join(", ")} · ${r.window_counts.test} overlapping windows. Participants, not windows, define the independent groups.`;
  const table = $("metrics-table");
  table.replaceChildren(
    tableRow(
      ["Model", "Accuracy", "Macro F1", "Coverage", "Accepted accuracy"],
      true,
    ),
  );
  for (const key of [
    "static_raw",
    "temporal_raw",
    "static_selective",
    "temporal_selective",
  ]) {
    const m = r.test[key];
    const label = {
      static_raw: "Mean pose",
      temporal_raw: "Pose + motion",
      static_selective: "Mean pose + rejection",
      temporal_selective: "Pose + motion + rejection",
    }[key];
    table.append(
      tableRow([
        label,
        percent(m.accuracy),
        percent(m.macro_f1),
        percent(m.coverage),
        percent(m.accepted_accuracy),
      ]),
    );
  }
  const m = r.test.temporal_selective,
    cm = $("confusion-table");
  cm.replaceChildren(
    tableRow(
      ["Actual / predicted", ...m.labels.map((l) => shortNames[l])],
      true,
    ),
  );
  m.confusion_matrix.slice(0, 6).forEach((row, i) => {
    const tr = tableRow([shortNames[m.labels[i]], ...row]);
    row.forEach((v, j) => {
      tr.children[j + 1].style.background = v
        ? `rgba(63,112,85,${0.08 + (0.3 * v) / Math.max(1, ...row)})`
        : "transparent";
    });
    cm.append(tr);
  });
  $("importance").replaceChildren();
  for (const [name, value] of r.feature_importance.slice(0, 6)) {
    const row = el("div", undefined, "feature-row");
    row.append(el("span", name.replaceAll("_", " ")));
    const bar = el("div", undefined, "feature-bar"),
      fill = el("i");
    fill.style.width = `${(value / r.feature_importance[0][1]) * 100}%`;
    bar.append(fill);
    row.append(bar, el("span", percent(value)));
    $("importance").append(row);
  }
}
async function train() {
  const ids = (id) =>
    $(id)
      .value.split(",")
      .map((x) => x.trim())
      .filter(Boolean);
  $("train").disabled = true;
  $("train-state").textContent =
    "Extracting windows, training two classifiers, and evaluating…";
  notice("");
  try {
    const r = await api("/api/train", {
      validation_subjects: ids("validation-subjects"),
      test_subjects: ids("test-subjects"),
    });
    renderResults(r);
    await refreshStatus();
  } catch (e) {
    $("train-state").textContent =
      "Experiment did not finish. Check the message above.";
    throw e;
  } finally {
    $("train").disabled = false;
  }
}
function exportSession() {
  if (!sessionLog.length)
    throw new Error(
      "Start a source and wait for predictions before exporting a session.",
    );
  const keys = Object.keys(sessionLog[0]),
    csv = [
      keys.join(","),
      ...sessionLog.map((r) => keys.map((k) => r[k]).join(",")),
    ].join("\n");
  const a = el("a");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.href = url;
  a.download = `poselab_${mode ?? "session"}_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
for (const b of document.querySelectorAll("[data-tab]"))
  b.addEventListener(
    "click",
    safe(async () => {
      document
        .querySelectorAll(".page,.nav-button")
        .forEach((e) => e.classList.remove("active"));
      b.classList.add("active");
      $(b.dataset.tab).classList.add("active");
      if (b.dataset.tab === "dataset") await refreshDataset();
      if (b.dataset.tab === "experiment")
        renderResults(await api("/api/results"));
    }),
  );
$("camera").addEventListener("click", safe(camera));
$("video-file").addEventListener("change", safe(openVideo));
$("demo").addEventListener("click", safe(simulation));
$("stop").addEventListener("click", stopSource);
$("record").addEventListener("click", safe(record));
$("train").addEventListener("click", safe(train));
$("refresh-data").addEventListener("click", safe(refreshDataset));
$("export-session").addEventListener("click", safe(exportSession));
$("reset-reps").addEventListener("click", () => {
  counter.reset();
  updateCounts();
});
$("engine").addEventListener("change", () => {
  generation++;
  busy = false;
  smoother.reset();
  counter.interrupt();
  frames = [];
  lastPrediction = -1;
  clearPrediction("Collecting a new movement window.");
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && mode) stopSource();
});
window.addEventListener("beforeunload", stopSource);
await safe(async () => {
  await refreshStatus();
  await refreshDataset();
  renderResults(await api("/api/results"));
})();
