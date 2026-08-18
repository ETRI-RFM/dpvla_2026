const SERVICES = [
  "image_server", "image_client", "brainco_hand",
  "gr00t_server", "recorder", "inference",
];
let config = null;
const streams = {};                   
let modelType = "gr00t";              
let mode = "eval";                   
let inferActive = "";
let inferActiveSource = "—";
let _shoulderUnlocked = false;        
let _lastStatus = {};                
let _inferenceReadyToRun = false;
const _RUN_PROMPT_RE = /Press 'Run inference' button/;
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
async function fetchJSON(path) {
  const r = await fetch(path);
  return r.json();
}
async function postJSON(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  return r.json();
}
function setStatus(text, kind = "ok") {
  $(".status-text").textContent = text;
  $(".dot").className = "dot " + kind;
}
let _instrTimer = null;
let _lastInstruction = null;
function writeInstructionFile(text) {
  if (text == null) return;
  if (text === _lastInstruction) return;
  _lastInstruction = text;
  if (_instrTimer) clearTimeout(_instrTimer);
  _instrTimer = setTimeout(async () => {
    try {
      const r = await postJSON("/api/instruction", {text});
      if (r.ok) updateSavedBadges(text);
      else setStatus(`instruction write failed: ${r.error}`, "warn");
    } catch (e) {
      setStatus("instruction write failed", "warn");
    }
  }, 80);
}
function updateSavedBadges(text) {
  const short = text.length > 40 ? text.slice(0, 37) + "…" : text;
  $$(".saved-badge").forEach(b => {
    b.textContent = `✓ saved → language_instruction.txt`;
    b.title = `${short}\n→ ${config?.lang_instruction_txt || ""}`;
  });
}
function setServiceState(label, state) {
  const el = $(`.service[data-label="${label}"] .state`);
  if (!el) return;
  el.textContent = state;
  el.dataset.state = state;
}
const _COLORED_SERVICES = new Set([
  "image_server", "brainco_hand", "image_client", "gr00t_server",
]);
const _AUTO_RUN_OK_SERVICES = new Set(["image_client"]);
const _READY_PATTERNS = {
  gr00t_server: /Server is ready and listening on tcp/,
};
const _serviceRunState = {};
function setServiceRunState(label, state) {
  if (!_COLORED_SERVICES.has(label)) return;
  if (_serviceRunState[label] === state) return;
  _serviceRunState[label] = state;
  const article = document.querySelector(`.service[data-label="${label}"]`);
  if (!article) return;
  article.classList.remove("run-ok", "run-stopped");
  if (state === "ok")   article.classList.add("run-ok");
  if (state === "stop") article.classList.add("run-stopped");
  refreshServiceButtons();
}
const ANSI_RE = new RegExp(
  "\\x1B(?:" +
    "\\[[0-?]*[ -/]*[@-~]" +                       
    "|\\][^\\x07\\x1B]*(?:\\x07|\\x1B\\\\)" +     
    "|[@-Z\\\\^_]" +                                
  ")", "g"
);
function stripAnsi(s) {
  return s.replace(ANSI_RE, "").replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
}
const LOG_MAX_CHARS = 200_000;
const LOG_TRIM_TO   = 100_000;
const _STOPPED_PATTERN = /\[exited with code|\[sent __INTR__\]|\[sent SIGINT|Traceback|^Error:|ERROR\b/;
function appendLog(label, line) {
  const el = $(`.service[data-label="${label}"] [data-role="log"]`);
  if (!el) return;
  const wasAtBottom =
    el.scrollHeight - el.clientHeight - el.scrollTop < 20;
  let text = el.textContent + stripAnsi(line);
  if (text.length > LOG_MAX_CHARS) {
    text = text.slice(-LOG_TRIM_TO);
  }
  el.textContent = text;
  if (wasAtBottom) el.scrollTop = el.scrollHeight;
  const readyRe = _READY_PATTERNS[label];
  if (readyRe && readyRe.test(line) && _serviceRunState[label] !== "ok") {
    setServiceRunState(label, "ok");
  }
  if (_COLORED_SERVICES.has(label)
      && _serviceRunState[label] === "ok"
      && _STOPPED_PATTERN.test(line)) {
    setServiceRunState(label, "stop");
  }
  if (label === "inference" && _RUN_PROMPT_RE.test(line)) {
    if (!_inferenceReadyToRun) {
      _inferenceReadyToRun = true;
      refreshInferenceButtons();
    }
  }
}
function clearLog(label) {
  const el = $(`.service[data-label="${label}"] [data-role="log"]`);
  if (el) el.textContent = "";
}
let _allStream = null;
function startAllStream() {
  if (_allStream) return;
  console.log("[sse] subscribe _all (multiplexed)");
  const evt = new EventSource("/events/_all");
  evt.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      appendLog(data.label, data.line);
    } catch {/* ignore */}
  };
  evt.onerror = () => {/* browser auto-reconnects */};
  _allStream = evt;
}
function startStream(_label) { startAllStream(); }
function stopStream(_label) { /* multiplexed → nothing to close */ }
let _camPreviewOn = false;
let _camPreviewStartedAt = 0;
let _refreshStatusCalls = 0;
async function refreshStatus() {
  _refreshStatusCalls++;
  try {
    const status = await fetchJSON("/api/status");
    _lastStatus = status;
    if (_refreshStatusCalls <= 3 || _refreshStatusCalls % 8 === 0) {
      console.log(`[refreshStatus #${_refreshStatusCalls}]`, status);
    }
    for (const label of SERVICES) {
      const info = status[label];
      if (!info) {
        setServiceState(label, "idle");
        stopStream(label);
      } else if (info.alive) {
        setServiceState(label, "running");
        startStream(label);
      } else {
        setServiceState(label, info.rc === 0 ? "exited" : "failed");
        stopStream(label);
      }
      if (_AUTO_RUN_OK_SERVICES.has(label)) {
        if (info && info.alive) {
          setServiceRunState(label, "ok");
        } else if (_serviceRunState[label] === "ok") {
          setServiceRunState(label, (info && info.rc === 0) ? null : "stop");
        }
      }
    }
    refreshServiceButtons();
    refreshInferenceButtons();
    const camAlive = !!(status.image_client && status.image_client.alive);
    if (camAlive && !_camPreviewOn) showCameraPreview();
    else if (!camAlive && _camPreviewOn) hideCameraPreview();
    if (_camPreviewOn) tickCameraPreviewOverlay();
  } catch {/* ignore */}
}
function refreshInferenceButtons() {
  const goBtn  = $("#btn-go-init");
  const runBtn = $("#btn-run-inference");
  if (!goBtn || !runBtn) return;
  const grootAlive = !!(_lastStatus?.gr00t_server?.alive);
  const needsGroot = modelType === "gr00t";
  const ok = _shoulderUnlocked && (!needsGroot || grootAlive);
  const infAlive = !!(_lastStatus?.inference?.alive);
  if (!infAlive && _inferenceReadyToRun) _inferenceReadyToRun = false;
  goBtn.disabled  = !ok || infAlive;
  runBtn.disabled = !ok || !infAlive || !_inferenceReadyToRun;
}
function wireSafetyGate() {
  const btn = $("#btn-shoulder-confirm");
  if (!btn) return;
  btn.addEventListener("click", () => {
    if (_shoulderUnlocked) return;
    _shoulderUnlocked = true;
    btn.disabled = true;
    btn.textContent = "✓ 확인 완료";
    $("#safety-gate")?.classList.add("confirmed");
    refreshInferenceButtons();
    setStatus("shoulder unlock confirmed", "ok");
  }, {once: true});
}
function refreshServiceButtons() {
  $$(".service").forEach(article => {
    const label = article.dataset.label;
    const state = article.querySelector(".state")?.dataset.state || "idle";
    const alive = state === "running";
    const colored = _COLORED_SERVICES.has(label);
    const runOk = colored && _serviceRunState[label] === "ok";
    article.querySelectorAll('button[data-action="start"]').forEach(b => {
      b.disabled = alive;
    });
    article.querySelectorAll('button[data-action="send"]').forEach(b => {
      b.disabled = !alive || runOk;
    });
    article.querySelectorAll('button[data-action="stop"]').forEach(b => {
      b.disabled = !alive;
    });
  });
}
function tickCameraPreviewOverlay() {
  const wrap = $("#camera-preview");
  if (!wrap || wrap.classList.contains("ready")) return;
  const overlay = wrap.querySelector(".preview-overlay");
  if (!overlay) return;
  const elapsed = (Date.now() - _camPreviewStartedAt) / 1000;
  if (elapsed > 8) {
    overlay.textContent =
      "no frames — is G1 Camera running? (① Connect + ▶ Run image_server.py)";
  } else if (elapsed > 3) {
    overlay.textContent = "waiting for first frame…";
  } else {
    overlay.textContent = "connecting…";
  }
}
let _camPollStop = null;
function showCameraPreview() {
  if (!config) return;
  const port = config.mjpeg_port || 8766;
  const wrap = $("#camera-preview");
  const img  = $("#camera-stream");
  const host = location.hostname || "127.0.0.1";
  const url  = `http://${host}:${port}/snapshot.jpg`;
  console.log("[preview] starting poll on", url);
  wrap.removeAttribute("hidden");
  wrap.style.display = "";
  img.onload  = () => { wrap.classList.add("ready"); };
  img.onerror = () => { /* swallow — interval will retry */ };
  img.src = `${url}?t=${Date.now()}`;
  const handle = setInterval(() => {
    img.src = `${url}?t=${Date.now()}`;
  }, 33);

  _camPollStop = () => clearInterval(handle);
  _camPreviewOn = true;
  _camPreviewStartedAt = Date.now();
}
function hideCameraPreview() {
  if (_camPollStop) { _camPollStop(); _camPollStop = null; }
  const wrap = $("#camera-preview");
  const img  = $("#camera-stream");
  wrap.hidden = true;
  wrap.classList.remove("ready");
  img.removeAttribute("src");
  _camPreviewOn = false;
}
async function svcStart(label, params = {}) {
  console.log(`[svcStart] → POST /api/service/start  label=${label}`, params);
  clearLog(label);
  let r;
  try {
    r = await postJSON("/api/service/start", {label, params});
    console.log(`[svcStart] ← response:`, r);
  } catch (e) {
    console.error(`[svcStart] POST failed:`, e);
    setStatus(`${label}: POST failed (${e.message})`, "err");
    return null;
  }
  if (!r.ok) {
    setStatus(`${label}: ${r.error}`, "warn");
  } else {
    setStatus(`${label} started (pid ${r.pid})`, "info");
  }
  refreshStatus();
  return r;
}
async function svcStop(label, mode = "kill") {
  const r = await postJSON("/api/service/stop", {label, mode});
  if (!r.ok) setStatus(`${label} stop: ${r.error}`, "warn");
  else setStatus(`${label} ${mode === "intr" ? "Ctrl+C" : "stop"} sent`, "info");
  refreshStatus();
  return r;
}
async function svcSend(label, text) {
  const r = await postJSON("/api/service/send", {label, text});
  if (!r.ok) setStatus(`${label} send: ${r.error}`, "warn");
  return r;
}
function wireServiceButtons() {
  const services = $$(".service");
  console.log(`[wire] found ${services.length} .service articles`);
  services.forEach(article => {
    const label = article.dataset.label;
    const btns = $$("button[data-action]", article);
    console.log(`[wire]   ${label}: ${btns.length} action buttons`);
    btns.forEach(btn => {
      btn.addEventListener("click", () => {
        const action = btn.dataset.action;
        const stopMode = btn.dataset.stopMode || "kill";
        const payload = btn.dataset.payload;
        const paramsKey = btn.dataset.params;
        console.log(
          `[svc-click] label=${label} action=${action} `
          + `paramsKey=${paramsKey || "—"} payload=${payload ? `"${payload}"` : "—"}`
        );
        const params = {};
        if (paramsKey === "model_path") {
          params.model_path = $("#model-path").value.trim();
        } else if (paramsKey === "rec_device") {
          const recDev = $("#rec-device");
          if (recDev) params.device = recDev.value.trim();
        }
        if (action === "start") {
          svcStart(label, params);
          setServiceRunState(label, null);
        } else if (action === "stop") {
          svcStop(label, stopMode);
          setServiceRunState(label, "stop");
        } else if (action === "send" && payload != null) {
          svcSend(label, payload);
          setServiceRunState(label, "ok");
        }
      });
    });
  });
}

function wireModelToggle() {
  $$("#model-type .seg").forEach(b => b.addEventListener("click", () => {
    $$("#model-type .seg").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    modelType = b.dataset.value;
    updateModelUI();
  }));
}

function updateModelUI() {
  const lbl  = $("#model-path-label");
  const path = $("#model-path");
  const gr00t = $("#gr00t-card");
  if (modelType === "gr00t") {
    lbl.textContent = "GR00T checkpoint";
    path.value = config.gr00t_default_model;
    gr00t.style.display = "";
  } else {
    lbl.textContent = "system1_cfg_path";
    path.value = config.dp_vla_default_model;
    gr00t.style.display = "none";
  }
  refreshInferenceButtons();
}


function wireModeToggle() {
  $$("#mode-type .seg").forEach(b => b.addEventListener("click", () => {
    $$("#mode-type .seg").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    mode = b.dataset.value;
    $("#eval-panel").hidden  = mode !== "eval";
    $("#infer-panel").hidden = mode !== "infer";
    if (mode === "eval") refreshEvalResolved();
    else if (inferActive) writeInstructionFile(inferActive);
  }));
}

function fillEvalTask() {
  const tasks = Object.keys(config.eval_protocol).sort();
  $("#eval-task").innerHTML =
    tasks.map(t => `<option value="${t}">${t}</option>`).join("");
  if (tasks.length) $("#eval-task").value = tasks[0];
  fillEvalEpisode();
}
function fillEvalEpisode() {
  const task = $("#eval-task").value;
  const eps = Object.keys(config.eval_protocol[task] || {})
    .sort((a, b) => +a - +b);
  $("#eval-episode").innerHTML =
    eps.map(e => `<option value="${e}">${e}</option>`).join("");
  if (eps.length) $("#eval-episode").value = eps[0];
  refreshEvalResolved();
}
function refreshEvalResolved() {
  const task = $("#eval-task").value;
  const ep   = $("#eval-episode").value;
  const type = $("input[name='eval-inst']:checked").value;
  const e = config.eval_protocol[task]?.[ep];
  if (e) {
    $("#eval-init-pose").textContent  = e.init_pose_idx;
    $("#eval-instruction").textContent = e[type];
    if (mode === "eval") writeInstructionFile(e[type]);
  } else {
    $("#eval-init-pose").textContent = "—";
    $("#eval-instruction").textContent = "—";
  }
}
function wireEval() {
  $("#eval-task").addEventListener("change", fillEvalEpisode);
  $("#eval-episode").addEventListener("change", refreshEvalResolved);
  $$("input[name='eval-inst']").forEach(el =>
    el.addEventListener("change", refreshEvalResolved));
}

function categoryOptions() {
  const opts = [];
  for (const sec of ["short", "long"]) {
    const cap = sec[0].toUpperCase() + sec.slice(1);
    const tasks = Object.keys(config.lang_lists[sec] || {})
      .sort((a, b) => +a - +b);
    for (const t of tasks) {
      opts.push({value: `${sec}|${t}`, label: `${cap} · Task ${t}`});
    }
  }
  return opts;
}
function fillInferCategory() {
  const opts = categoryOptions();
  $("#infer-category").innerHTML =
    opts.map(o => `<option value="${o.value}">${o.label}</option>`).join("");
  if (opts.length) $("#infer-category").value = opts[0].value;
  fillInferInstruction();
}
function fillInferInstruction() {
  const cat = $("#infer-category").value;
  if (!cat) { $("#infer-instruction").innerHTML = ""; return; }
  const [sec, t] = cat.split("|");
  const lines = (config.lang_lists[sec]?.[t] || [])
    .slice()
    .sort((a, b) => a.localeCompare(b, undefined, {sensitivity: "base", numeric: true}));
  $("#infer-instruction").innerHTML =
    `<option value="">(pick one)</option>` +
    lines.map(l => {
      const safe = l.replace(/"/g, "&quot;").replace(/</g, "&lt;");
      return `<option value="${safe}">${safe}</option>`;
    }).join("");
}
function setInferActive(text, source) {
  inferActive = text;
  inferActiveSource = source;
  $("#infer-active-source").textContent = source;
  $("#infer-active").textContent = text || "—";
  if (mode === "infer" && text) writeInstructionFile(text);
}
function wireInfer() {
  $("#infer-category").addEventListener("change", fillInferInstruction);
  $("#infer-instruction").addEventListener("change", () => {
    const v = $("#infer-instruction").value;
    if (v) {
      const cat = $("#infer-category");
      const label = cat.selectedOptions[0]?.textContent || "Inference";
      setInferActive(v, label);
    }
  });
  $("#infer-send-custom").addEventListener("click", () => {
    const text = $("#infer-custom").value.trim();
    if (!text) { setStatus("custom instruction empty", "warn"); return; }
    setInferActive(text, "Custom");
    setStatus("custom instruction set", "info");
  });
}
function resolveRunArgs() {
  if (mode === "eval") {
    const task = $("#eval-task").value;
    const ep   = $("#eval-episode").value;
    const type = $("input[name='eval-inst']:checked").value;
    const e = config.eval_protocol[task]?.[ep];
    if (!e) { setStatus("pick task & episode", "warn"); return null; }
    return {episode_idx: e.init_pose_idx, instruction: e[type]};
  }
  const v = parseInt($("#infer-init-pose").value, 10);
  if (isNaN(v)) {
    setStatus("init_pose_idx required", "warn"); return null;
  }
  if (v < config.init_pose_min || v > config.init_pose_max) {
    setStatus(`init_pose_idx out of range (${config.init_pose_min}..${config.init_pose_max})`, "err");
    return null;
  }
  if (!inferActive) {
    setStatus("no active instruction (pick or Send custom)", "warn"); return null;
  }
  return {episode_idx: v, instruction: inferActive};
}

function wireInferenceButtons() {
  $("#btn-go-init").addEventListener("click", async () => {
    const args = resolveRunArgs();
    if (!args) return;
    $("#btn-go-init").disabled = true;
    _inferenceReadyToRun = false;
    const params = {
      model_type: modelType,
      model_path: $("#model-path").value.trim(),
      episode_idx: args.episode_idx,
      instruction: args.instruction,
    };
    const r = await svcStart("inference", params);
    if (r.ok) {
      const short = args.instruction.length > 80
        ? args.instruction.slice(0, 77) + "…" : args.instruction;
      $("#summary").textContent = `instruction: ${short}`;
      setStatus("inference launching — wait for prompt", "info");
    } else {
      refreshInferenceButtons();
    }
  });
  $("#btn-run-inference").addEventListener("click", async () => {
    _inferenceReadyToRun = false;
    refreshInferenceButtons();
    const r = await svcSend("inference", "s");
    if (r.ok) setStatus("inference running", "ok");
  });
  $("#btn-finish").addEventListener("click", async () => {
    _inferenceReadyToRun = false;
    refreshInferenceButtons();
    const r = await svcStop("inference", "kill");
    if (r.ok) setStatus("stopping…", "warn");
  });
}

let _recPreviewOn = false;
let _recPollStop = null;
let _recStatusTimer = null;

const REC_SNAPSHOT_URL = "/api/recorder/snapshot.jpg";
const REC_STATUS_URL   = "/api/recorder/status";
const REC_START_URL    = "/api/recorder/start";
const REC_STOP_URL     = "/api/recorder/stop";

function showRecorderPreview() {
  if (!config) return;
  const wrap = $("#rec-preview");
  const img  = $("#rec-stream");
  const overlay = wrap.querySelector(".preview-overlay");
  wrap.removeAttribute("hidden");
  let stopped = false;
  let okCount = 0, errCount = 0;
  let lastErr = "";
  const t0 = Date.now();
  console.log("[recorder] preview start (fetch+blob v4 — deferred revoke)");

  const updateDebug = () => {
    if (overlay && !wrap.classList.contains("ready")) {
      const s = ((Date.now() - t0) / 1000).toFixed(1);
      overlay.textContent =
        `fetch+blob v4  ·  ok=${okCount} err=${errCount}  ·  ${s}s` +
        (lastErr ? `  ·  ${lastErr}` : "");
    }
  };

  let prevBlobUrl = null;
  function setImg(newUrl) {
    const toRevoke = prevBlobUrl;
    prevBlobUrl = newUrl;
    img.onload = () => {
      if (toRevoke) URL.revokeObjectURL(toRevoke);
      if (!wrap.classList.contains("ready")) wrap.classList.add("ready");
    };
    img.src = newUrl;
  }

  async function tick() {
    let iter = 0;
    while (!stopped) {
      iter++;
      try {
        const r = await fetch(`${REC_SNAPSHOT_URL}?t=${Date.now()}`,
          {cache: "no-store"});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const blob = await r.blob();
        if (stopped) return;
        setImg(URL.createObjectURL(blob));
        okCount++;
        if (iter === 1) console.log("[recorder] first frame OK");
        await new Promise(res => setTimeout(res, 33)); 
      } catch (e) {
        errCount++;
        lastErr = e.message || String(e);
        console.warn(`[recorder] iter ${iter}: ${lastErr}`);
        updateDebug();
        await new Promise(res => setTimeout(res, 250));  
      }
    }
    console.log(`[recorder] tick exited after ${iter} iterations`);
  }

  tick();
  setTimeout(updateDebug, 800);
  _recPollStop = () => {
    stopped = true;
    if (prevBlobUrl) {
      URL.revokeObjectURL(prevBlobUrl);
      prevBlobUrl = null;
    }
  };
  _recPreviewOn = true;
}

function hideRecorderPreview() {
  if (_recPollStop) { _recPollStop(); _recPollStop = null; }
  const wrap = $("#rec-preview");
  const img  = $("#rec-stream");
  wrap.hidden = true;
  wrap.classList.remove("ready");
  img.removeAttribute("src");
  _recPreviewOn = false;
}

async function recGetStatus() {
  try {
    const r = await fetch(REC_STATUS_URL, {cache: "no-store"});
    return await r.json();
  } catch {
    return null;
  }
}

function formatDuration(s) {
  s = Math.floor(s);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

async function refreshRecorderStatus() {
  const el      = $("#rec-status");
  const btnRec  = $("#btn-rec-start");
  const btnStop = $("#btn-rec-stop");
  const wrap    = $("#rec-preview");
  if (!el) return;
  const s = await recGetStatus();
  if (!s) {
    el.className = "rec-status";
    el.textContent =
      "recorder offline — device busy? restart server or close other apps using /dev/video4";
    if (wrap && !wrap.classList.contains("ready")) {
      const ov = wrap.querySelector(".preview-overlay");
      if (ov) ov.textContent = "recorder offline";
    }
    if (btnRec)  btnRec.disabled  = true;
    if (btnStop) btnStop.disabled = true;
    return;
  }
  if (s.recording) {
    el.className = "rec-status recording";
    const dur = formatDuration(s.duration_s || 0);
    const sz  = s.size ? `${s.size[0]}x${s.size[1]}` : "";
    el.textContent =
      `REC  ${dur}   ${s.frames} frames   ${sz}   → ${s.file}`;
    if (btnRec)  btnRec.disabled  = true;
    if (btnStop) btnStop.disabled = false;
  } else {
    el.className = "rec-status";
    el.textContent = "ready (not recording)";
    if (btnRec)  btnRec.disabled  = false;
    if (btnStop) btnStop.disabled = true;
  }
}

function wireRecorder() {
  $("#btn-rec-start").addEventListener("click", async () => {
    const body = {
      out_dir:  $("#rec-out-dir")?.value.trim()  || "",
      filename: $("#rec-filename")?.value.trim() || "",
    };
    const r = await postJSON(REC_START_URL, body);
    if (!r.ok) {
      setStatus(`record: ${r.error}`, "warn");
      return;
    }
    setStatus(`recording → ${r.file}`, "info");
    refreshRecorderStatus();
  });

  $("#btn-rec-stop").addEventListener("click", async () => {
    const r = await postJSON(REC_STOP_URL, {});
    if (!r.ok) {
      setStatus(`stop record: ${r.error}`, "warn");
      return;
    }
    const el = $("#rec-status");
    if (el) {
      el.className = "rec-status done";
      el.textContent =
        `saved · ${r.frames} frames · ${formatDuration(r.duration_s)} → ${r.file}`;
    }
    setStatus(`saved → ${r.file}`, "ok");
  });

  wireVideoPlayer();

  const browseBtn = $("#rec-out-browse");
  if (browseBtn) {
    browseBtn.addEventListener("click", () => {
      const cur = $("#rec-out-dir").value.trim();
      const start = cur || config?.recorder_out_dir || "~/Videos";
      fsOpen(start, (picked) => {
        $("#rec-out-dir").value = picked;
      });
    });
  }
}

let _videoBlobUrl = null;
function wireVideoPlayer() {
  const fileInput = $("#video-file-input");
  const pickBtn   = $("#video-pick-file");
  const player    = $("#video-player");
  const errEl     = $("#video-error");
  if (!fileInput || !pickBtn || !player) return;

  pickBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    if (errEl) errEl.hidden = true;
    if (_videoBlobUrl) {
      URL.revokeObjectURL(_videoBlobUrl);
      _videoBlobUrl = null;
    }
    _videoBlobUrl = URL.createObjectURL(f);
    player.src = _videoBlobUrl;
    player.load();
    player.play().catch(() => {/* user-gesture or unsupported codec */});
    setStatus(`loaded ${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)`, "info");
  });

  player.addEventListener("error", () => {
    const e = player.error;
    const reasons = {
      1: "fetch aborted",
      2: "network error",
      3: "decode failed (codec not supported by browser — try H.264 mp4)",
      4: "src not supported (need .mp4 / .webm / .mov)",
    };
    const msg = reasons[e?.code] || `error ${e?.code}`;
    if (errEl) {
      errEl.hidden = false;
      errEl.textContent = `❌ ${msg}`;
      errEl.style.color = "var(--danger)";
    }
    console.warn("[video] error:", e);
  });
}

let _fsOnSelect = null;
let _fsCurrent = null;

async function fsOpen(initialPath, onSelect) {
  _fsOnSelect = onSelect;
  await fsNavigate(initialPath);
  $("#fs-modal").removeAttribute("hidden");
}
function fsClose() {
  $("#fs-modal").setAttribute("hidden", "");
  _fsOnSelect = null;
}

async function fsNavigate(target) {
  const list = $("#fs-list");
  list.innerHTML = '<div class="fs-empty">loading…</div>';
  const r = await postJSON("/api/fs/list", {path: target});
  if (!r.ok) {
    list.innerHTML = `<div class="fs-error">${r.error}</div>`;
    return;
  }
  _fsCurrent = r.path;
  $("#fs-current-path").textContent = r.path;
  $("#fs-path-input").value = r.path;
  list.innerHTML = "";
  if (r.dirs.length === 0 && r.files.length === 0) {
    list.innerHTML = '<div class="fs-empty">(empty directory)</div>';
    return;
  }
  for (const d of r.dirs) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="icon">📁</span><span></span>`;
    li.querySelector("span:last-child").textContent = d.name;
    li.addEventListener("click", () => fsNavigate(d.path));
    list.appendChild(li);
  }
  for (const f of r.files) {
    const li = document.createElement("li");
    li.className = "is-file";
    li.innerHTML = `<span class="icon">📄</span><span></span>`;
    li.querySelector("span:last-child").textContent = f.name;
    list.appendChild(li);
  }
}

function wireFsModal() {

  $$("#fs-modal [data-close]").forEach(el =>
    el.addEventListener("click", fsClose));

  $("#fs-up").addEventListener("click", async () => {
    if (!_fsCurrent) return;
    const idx = _fsCurrent.lastIndexOf("/");
    const parent = idx > 0 ? _fsCurrent.slice(0, idx) : "/";
    fsNavigate(parent);
  });
  $("#fs-home").addEventListener("click", () => fsNavigate("~"));
  $("#fs-go").addEventListener("click", () =>
    fsNavigate($("#fs-path-input").value.trim()));
  $("#fs-path-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      fsNavigate($("#fs-path-input").value.trim());
    }
  });
  $("#fs-select").addEventListener("click", () => {
    if (_fsOnSelect && _fsCurrent) _fsOnSelect(_fsCurrent);
    fsClose();
  });


  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#fs-modal").hasAttribute("hidden")) {
      fsClose();
    }
  });

  $("#model-path-browse").addEventListener("click", () => {
    const cur = $("#model-path").value.trim();
    let start;
    if (cur) {

      const idx = cur.lastIndexOf("/");
      start = idx > 0 ? cur.slice(0, idx) : cur;
    } else {
      start = modelType === "gr00t"
        ? "/mnt/ssd/GROOT"
        : "/home/goodman/unitree_v030/act_validation";
    }
    fsOpen(start, (picked) => {
      $("#model-path").value = picked;
    });
  });
}

function wireShutdown() {
  $("#shutdown-btn").addEventListener("click", async () => {
    if (!confirm("Shutdown the server and kill all subprocesses?")) return;
    try { await postJSON("/api/shutdown", {}); } catch {}
    document.body.innerHTML = `
      <main style="padding:40px;text-align:center">
        <h1 style="font-family:Ubuntu,sans-serif;color:#1F2937">Shut down.</h1>
        <p style="color:#6B7280">All subprocesses terminated. You can close this tab.</p>
      </main>`;
  });
}

async function init() {
  console.log("[init] start");

  document.addEventListener("click", (e) => {
    const t = e.target.closest("button, .seg, [data-close], a, input, select");
    if (t) {
      console.log(
        "[click]",
        t.tagName,
        t.id || "",
        t.className || "",
        "data-action=", t.dataset?.action || "—",
      );
    }
  }, true);

  config = await fetchJSON("/api/config");
  console.log("[init] config loaded:", Object.keys(config));
  $("#model-path").value = config.gr00t_default_model;
  const recDirInput = $("#rec-out-dir");
  if (recDirInput) recDirInput.value = config.recorder_out_dir || "~/Videos";

  try { fillEvalTask();          console.log("[init] ✓ fillEvalTask"); }
    catch (e) { console.error("[init] ✗ fillEvalTask:", e); }
  try { fillInferCategory();     console.log("[init] ✓ fillInferCategory"); }
    catch (e) { console.error("[init] ✗ fillInferCategory:", e); }
  try { wireServiceButtons();    console.log("[init] ✓ wireServiceButtons"); }
    catch (e) { console.error("[init] ✗ wireServiceButtons:", e); }
  try { wireModelToggle();       console.log("[init] ✓ wireModelToggle"); }
    catch (e) { console.error("[init] ✗ wireModelToggle:", e); }
  try { wireModeToggle();        console.log("[init] ✓ wireModeToggle"); }
    catch (e) { console.error("[init] ✗ wireModeToggle:", e); }
  try { wireEval();              console.log("[init] ✓ wireEval"); }
    catch (e) { console.error("[init] ✗ wireEval:", e); }
  try { wireInfer();             console.log("[init] ✓ wireInfer"); }
    catch (e) { console.error("[init] ✗ wireInfer:", e); }
  try { wireInferenceButtons();  console.log("[init] ✓ wireInferenceButtons"); }
    catch (e) { console.error("[init] ✗ wireInferenceButtons:", e); }
  try { wireRecorder();          console.log("[init] ✓ wireRecorder"); }
    catch (e) { console.error("[init] ✗ wireRecorder:", e); }
  try { wireShutdown();          console.log("[init] ✓ wireShutdown"); }
    catch (e) { console.error("[init] ✗ wireShutdown:", e); }
  try { wireFsModal();           console.log("[init] ✓ wireFsModal"); }
    catch (e) { console.error("[init] ✗ wireFsModal:", e); }
  try { wireSafetyGate();        console.log("[init] ✓ wireSafetyGate"); }
    catch (e) { console.error("[init] ✗ wireSafetyGate:", e); }

  startAllStream();

  refreshStatus();
  setInterval(refreshStatus, 2500);

  showRecorderPreview();
  refreshRecorderStatus();
  _recStatusTimer = setInterval(refreshRecorderStatus, 1000);

  console.log("[init] complete — all wires done");
}
init().catch(err => {
  console.error(err);
  setStatus("init failed: " + err.message, "err");
});
