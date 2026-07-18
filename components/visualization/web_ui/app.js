const state = {
  versions: null,
  counts: { realtime: 0, change: 0, change_memory_objects: 0, speech_timeline: 0 },
  realtimeIndex: 0,
  changeIndex: 0,
  selectedObjectId: "",
  activeLogKind: "frame",
  lastMeshPayload: null,
  lastMeshInteractionAt: 0,
  meshUserInteracting: false,
};

const ui = {
  statusLine: document.getElementById("statusLine"),
  metaRealtime: document.getElementById("metaRealtime"),
  metaChange: document.getElementById("metaChange"),
  metaObjects: document.getElementById("metaObjects"),
  metaSpeech: document.getElementById("metaSpeech"),
  rtSlider: document.getElementById("rtSlider"),
  rtPrev: document.getElementById("rtPrev"),
  rtNext: document.getElementById("rtNext"),
  rtIndex: document.getElementById("rtIndex"),
  rtTimestamp: document.getElementById("rtTimestamp"),
  rtRgb: document.getElementById("rtRgb"),
  rtDepth: document.getElementById("rtDepth"),
  rtConf: document.getElementById("rtConf"),
  rtDescribe: document.getElementById("rtDescribe"),
  chSlider: document.getElementById("chSlider"),
  chPrev: document.getElementById("chPrev"),
  chNext: document.getElementById("chNext"),
  chIndex: document.getElementById("chIndex"),
  chCaption: document.getElementById("chCaption"),
  chCurrent: document.getElementById("chCurrent"),
  chReference: document.getElementById("chReference"),
  chGeminiCurrent: document.getElementById("chGeminiCurrent"),
  chGeminiRef: document.getElementById("chGeminiRef"),
  chMaskCurrent: document.getElementById("chMaskCurrent"),
  chMaskRef: document.getElementById("chMaskRef"),
  speechTimeline: document.getElementById("speechTimeline"),
  meshCanvas: document.getElementById("meshCanvas"),
  meshCaption: document.getElementById("meshCaption"),
  memoryList: document.getElementById("memoryList"),
  memoryDetail: document.getElementById("memoryDetail"),
  logsArea: document.getElementById("logsArea"),
  logTabButtons: Array.from(document.querySelectorAll(".log-tabs button")),
};

const threeCtx = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  fallbackControls: null,
  meshGroup: null,
  bboxGroup: null,
  grid: null,
  animating: false,
};

async function fetchJson(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function setImage(el, src) {
  if (!el) return;
  if (src) {
    el.src = src;
    el.style.visibility = "visible";
  } else {
    el.removeAttribute("src");
    el.style.visibility = "hidden";
  }
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function clampIndex(index, count) {
  if (count <= 0) return 0;
  return Math.max(0, Math.min(index, count - 1));
}

function updateChipCounts(counts) {
  ui.metaRealtime.textContent = `Realtime ${counts.realtime || 0}`;
  ui.metaChange.textContent = `Changes ${counts.change || 0}`;
  ui.metaObjects.textContent = `Objects ${counts.change_memory_objects || 0}`;
  ui.metaSpeech.textContent = `Speech ${counts.speech_timeline || 0}`;
}

function updateSliders() {
  const rtCount = state.counts.realtime || 0;
  ui.rtSlider.max = Math.max(0, rtCount - 1);
  state.realtimeIndex = clampIndex(state.realtimeIndex, rtCount);
  ui.rtSlider.value = state.realtimeIndex;
  ui.rtIndex.textContent = rtCount > 0 ? `${state.realtimeIndex + 1} / ${rtCount}` : "0 / 0";

  const chCount = state.counts.change || 0;
  ui.chSlider.max = Math.max(0, chCount - 1);
  state.changeIndex = clampIndex(state.changeIndex, chCount);
  ui.chSlider.value = state.changeIndex;
  ui.chIndex.textContent = chCount > 0 ? `${state.changeIndex + 1} / ${chCount}` : "0 / 0";
}

async function loadRealtime(index) {
  const payload = await fetchJson(`/api/realtime?index=${encodeURIComponent(index)}`);
  if (!payload || payload.count === undefined) return;
  state.realtimeIndex = payload.index || 0;
  updateSliders();
  setImage(ui.rtRgb, payload.rgb || "");
  setImage(ui.rtDepth, payload.depth || "");
  setImage(ui.rtConf, payload.confidence || "");
  const describeImage = payload.live_describing_image || payload.describing_image || "";
  setImage(ui.rtDescribe, describeImage);
  ui.rtTimestamp.textContent = payload.timestamp ? `Timestamp: ${payload.timestamp}` : "No realtime frame.";
}

async function loadChange(index) {
  const payload = await fetchJson(`/api/change?index=${encodeURIComponent(index)}`);
  if (!payload || payload.count === undefined) return;
  state.changeIndex = payload.index || 0;
  updateSliders();
  const panels = payload.panels || {};
  setImage(ui.chCurrent, panels.current_frame || "");
  setImage(ui.chReference, panels.reference_frame || "");
  setImage(ui.chGeminiCurrent, panels.current_gemini || "");
  setImage(ui.chGeminiRef, panels.reference_gemini || "");
  setImage(ui.chMaskCurrent, panels.current_mask || "");
  setImage(ui.chMaskRef, panels.reference_mask || "");
  const desc = payload.description || "";
  const ts = payload.timestamp || "";
  ui.chCaption.textContent = ts ? `${ts} | ${desc}` : desc || "No change event.";
}

function createMemoryItem(row) {
  const node = document.createElement("div");
  node.className = "memory-item";
  if (row.object_id === state.selectedObjectId) node.classList.add("active");
  node.dataset.objectId = row.object_id;
  node.innerHTML = `
    <div><strong>${row.object_id}</strong> <span class="${row.status === "disappeared" ? "warn" : ""}">[${row.status}]</span></div>
    <div class="tiny">${row.latest_change_type || ""} | snapshots: ${row.snapshot_count || 0}</div>
    <div class="tiny">${row.latest_timestamp || ""}</div>
    <div class="tiny">${escapeHtml(row.latest_description || "")}</div>
  `;
  node.addEventListener("click", async () => {
    state.selectedObjectId = row.object_id;
    await loadMemorySummary();
    await loadObjectDetail(row.object_id);
  });
  return node;
}

async function loadMemorySummary() {
  const rows = await fetchJson("/api/change-memory");
  if (!Array.isArray(rows)) return;
  if (rows.length > 0) {
    const exists = rows.some((row) => row.object_id === state.selectedObjectId);
    if (!exists) state.selectedObjectId = rows[0].object_id;
  }

  ui.memoryList.innerHTML = "";
  if (rows.length === 0) {
    ui.memoryList.innerHTML = `<p class="caption">No object in change memory yet.</p>`;
    ui.memoryDetail.innerHTML = "";
    return;
  }
  rows.forEach((row) => ui.memoryList.appendChild(createMemoryItem(row)));
  if (state.selectedObjectId) {
    await loadObjectDetail(state.selectedObjectId);
  }
}

function featureSection(label, payload) {
  if (!payload) return "";
  return `
    <details>
      <summary>${label}</summary>
      <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
    </details>
  `;
}

async function loadObjectDetail(objectId) {
  if (!objectId) return;
  const detail = await fetchJson(`/api/object?id=${encodeURIComponent(objectId)}`);
  if (!detail || !Array.isArray(detail.snapshots)) return;

  const head = `
    <div class="snapshot-card">
      <div class="snapshot-head">
        <strong>${escapeHtml(detail.object_id || "")}</strong>
        <span class="${detail.status === "disappeared" ? "warn" : ""}">${escapeHtml(detail.status || "")}</span>
      </div>
      <div class="snapshot-desc">${escapeHtml(detail.latest_description || "")}</div>
      <div class="tiny">${escapeHtml(detail.latest_timestamp || "")}</div>
    </div>
  `;

  const items = detail.snapshots
    .map((snap, idx) => {
      const img = snap.bbox_image ? `<img src="${snap.bbox_image}" alt="Snapshot image ${idx}" />` : "";
      return `
        <article class="snapshot-card">
          <div class="snapshot-head">
            <strong>${idx + 1}. ${escapeHtml(snap.change_type || "")}</strong>
            <span>${escapeHtml(snap.timestamp || "")}</span>
          </div>
          <div class="snapshot-desc">${escapeHtml(snap.description || "")}</div>
          ${img}
          <details>
            <summary>3D Bounding Box</summary>
            <pre>${escapeHtml(JSON.stringify(snap.bbox_3d || [], null, 2))}</pre>
          </details>
          ${featureSection("Visual Feature (DINO)", snap.dino_feature)}
          ${featureSection("Text Feature", snap.description_embedding)}
        </article>
      `;
    })
    .join("");

  ui.memoryDetail.innerHTML = head + items;
}

function initThree() {
  if (!window.THREE) {
    ui.meshCaption.textContent = "Three.js not loaded.";
    return;
  }
  if (threeCtx.renderer) return;

  const rect = ui.meshCanvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(240, Math.floor(rect.height));

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(width, height);
  if ("outputColorSpace" in renderer) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  } else if ("outputEncoding" in renderer) {
    renderer.outputEncoding = THREE.sRGBEncoding;
  }
  ui.meshCanvas.innerHTML = "";
  ui.meshCanvas.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5fbff);

  const camera = new THREE.PerspectiveCamera(58, width / height, 0.01, 1000);
  camera.position.set(1.4, 1.1, 1.4);
  camera.up.set(0, 1, 0);

  let controls = null;
  try {
    const OrbitCtor = THREE.OrbitControls || window.OrbitControls;
    if (OrbitCtor) {
      controls = new OrbitCtor(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(0, 0, 0);
      controls.addEventListener("start", () => {
        state.meshUserInteracting = true;
        markMeshInteraction();
      });
      controls.addEventListener("end", () => {
        state.meshUserInteracting = false;
        markMeshInteraction();
      });
    }
  } catch {
    controls = null;
  }

  const ambient = new THREE.AmbientLight(0xffffff, 0.68);
  scene.add(ambient);
  const mainLight = new THREE.DirectionalLight(0xffffff, 0.72);
  mainLight.position.set(3.0, 4.0, 2.0);
  scene.add(mainLight);
  const fillLight = new THREE.DirectionalLight(0xeaf6ff, 0.30);
  fillLight.position.set(-3.0, 1.6, -2.5);
  scene.add(fillLight);

  const grid = new THREE.GridHelper(4.0, 20, 0xb9cfdf, 0xd9e7f2);
  grid.position.y = -0.001;
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.45));

  const meshGroup = new THREE.Group();
  const bboxGroup = new THREE.Group();
  scene.add(meshGroup);
  scene.add(bboxGroup);

  const fallbackControls = controls ? null : createFallbackControls(camera, renderer.domElement);

  threeCtx.renderer = renderer;
  threeCtx.scene = scene;
  threeCtx.camera = camera;
  threeCtx.controls = controls;
  threeCtx.fallbackControls = fallbackControls;
  threeCtx.meshGroup = meshGroup;
  threeCtx.bboxGroup = bboxGroup;
  threeCtx.grid = grid;

  function animate() {
    if (!threeCtx.animating) return;
    requestAnimationFrame(animate);
    if (threeCtx.controls) {
      threeCtx.controls.update();
    } else if (threeCtx.fallbackControls) {
      threeCtx.fallbackControls.update();
    }
    renderer.render(scene, camera);
  }
  threeCtx.animating = true;
  animate();

  const resize = () => {
    if (!threeCtx.renderer || !threeCtx.camera) return;
    const box = ui.meshCanvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(box.width));
    const h = Math.max(220, Math.floor(box.height));
    threeCtx.renderer.setSize(w, h);
    threeCtx.camera.aspect = w / h;
    threeCtx.camera.updateProjectionMatrix();
  };
  window.addEventListener("resize", resize);

  if (!controls) {
    ui.meshCaption.textContent = "OrbitControls unavailable; using fallback drag/zoom controls.";
  }
}

function createFallbackControls(camera, canvas) {
  const target = new THREE.Vector3(0, 0, 0);
  const stateCtl = { theta: 0.78, phi: 1.02, radius: 2.4, dragging: false, lastX: 0, lastY: 0, changed: true };

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function apply() {
    const r = stateCtl.radius;
    const sinPhi = Math.sin(stateCtl.phi);
    camera.position.set(
      target.x + r * sinPhi * Math.cos(stateCtl.theta),
      target.y + r * Math.cos(stateCtl.phi),
      target.z + r * sinPhi * Math.sin(stateCtl.theta),
    );
    camera.lookAt(target);
  }

  canvas.addEventListener("pointerdown", (ev) => {
    state.meshUserInteracting = true;
    markMeshInteraction();
    stateCtl.dragging = true;
    stateCtl.lastX = ev.clientX;
    stateCtl.lastY = ev.clientY;
    canvas.setPointerCapture?.(ev.pointerId);
  });
  canvas.addEventListener("pointerup", (ev) => {
    state.meshUserInteracting = false;
    markMeshInteraction();
    stateCtl.dragging = false;
    canvas.releasePointerCapture?.(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!stateCtl.dragging) return;
    const dx = ev.clientX - stateCtl.lastX;
    const dy = ev.clientY - stateCtl.lastY;
    stateCtl.lastX = ev.clientX;
    stateCtl.lastY = ev.clientY;
    stateCtl.theta -= dx * 0.0085;
    stateCtl.phi = clamp(stateCtl.phi + dy * 0.0085, 0.10, Math.PI - 0.10);
    stateCtl.changed = true;
  });
  canvas.addEventListener(
    "wheel",
    (ev) => {
      ev.preventDefault();
      markMeshInteraction();
      const factor = ev.deltaY > 0 ? 1.08 : 0.92;
      stateCtl.radius = clamp(stateCtl.radius * factor, 0.20, 120.0);
      stateCtl.changed = true;
    },
    { passive: false },
  );

  return {
    target,
    setTarget: (x, y, z) => {
      target.set(x, y, z);
      stateCtl.changed = true;
    },
    setRadius: (radius) => {
      stateCtl.radius = clamp(radius, 0.20, 120.0);
      stateCtl.changed = true;
    },
    update: () => {
      if (!stateCtl.changed) return;
      apply();
      stateCtl.changed = false;
    },
  };
}

function markMeshInteraction() {
  state.lastMeshInteractionAt = Date.now();
}

function disposeGroup(group) {
  if (!group) return;
  while (group.children.length > 0) {
    const obj = group.children[0];
    if (!obj) continue;
    group.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
      else obj.material.dispose();
    }
  }
}

function remapPoint(raw) {
  const x = Number(raw?.[0] || 0);
  const y = Number(raw?.[1] || 0);
  const z = Number(raw?.[2] || 0);
  return [x, y, z];
}

function colorFromArray(raw, fallback = [0.15, 0.55, 0.85]) {
  const c = Array.isArray(raw) && raw.length >= 3 ? raw : fallback;
  return new THREE.Color(
    Math.max(0, Math.min(1, Number(c[0]))),
    Math.max(0, Math.min(1, Number(c[1]))),
    Math.max(0, Math.min(1, Number(c[2]))),
  );
}

function buildMeshObject(mesh) {
  const vertices = Array.isArray(mesh?.vertices) ? mesh.vertices : [];
  const triangles = Array.isArray(mesh?.triangles) ? mesh.triangles : [];
  if (vertices.length === 0 || triangles.length === 0) return null;

  const positions = new Float32Array(vertices.length * 3);
  vertices.forEach((vertex, i) => {
    const mapped = remapPoint(vertex);
    positions[i * 3 + 0] = mapped[0];
    positions[i * 3 + 1] = mapped[1];
    positions[i * 3 + 2] = mapped[2];
  });

  const indices = new Uint32Array(triangles.length * 3);
  triangles.forEach((tri, i) => {
    indices[i * 3 + 0] = Number(tri?.[0] || 0);
    indices[i * 3 + 1] = Number(tri?.[1] || 0);
    indices[i * 3 + 2] = Number(tri?.[2] || 0);
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  const colorsRaw = Array.isArray(mesh?.vertex_colors) ? mesh.vertex_colors : [];
  const hasColors = colorsRaw.length === vertices.length;
  if (hasColors) {
    const colors = new Float32Array(vertices.length * 3);
    colorsRaw.forEach((c, i) => {
      colors[i * 3 + 0] = Number(c?.[0] || 0);
      colors[i * 3 + 1] = Number(c?.[1] || 0);
      colors[i * 3 + 2] = Number(c?.[2] || 0);
    });
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    color: hasColors ? undefined : new THREE.Color(0x4d9bd4),
    vertexColors: hasColors,
    roughness: 0.84,
    metalness: 0.04,
    transparent: true,
    opacity: 0.92,
    side: THREE.DoubleSide,
  });
  return new THREE.Mesh(geometry, material);
}

function buildBBoxObject(box) {
  const minRaw = Array.isArray(box?.min) ? box.min : [0, 0, 0];
  const maxRaw = Array.isArray(box?.max) ? box.max : [0, 0, 0];
  const min = remapPoint(minRaw);
  const max = remapPoint(maxRaw);

  const corners = [
    [min[0], min[1], min[2]],
    [max[0], min[1], min[2]],
    [max[0], max[1], min[2]],
    [min[0], max[1], min[2]],
    [min[0], min[1], max[2]],
    [max[0], min[1], max[2]],
    [max[0], max[1], max[2]],
    [min[0], max[1], max[2]],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];

  const positions = new Float32Array(edges.length * 2 * 3);
  let ptr = 0;
  edges.forEach(([a, b]) => {
    const p1 = corners[a];
    const p2 = corners[b];
    positions[ptr++] = p1[0];
    positions[ptr++] = p1[1];
    positions[ptr++] = p1[2];
    positions[ptr++] = p2[0];
    positions[ptr++] = p2[1];
    positions[ptr++] = p2[2];
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: colorFromArray(box?.color), linewidth: 2 });
  return new THREE.LineSegments(geometry, material);
}

function focusCameraFromBounds() {
  if (!threeCtx.camera || !threeCtx.meshGroup || !threeCtx.bboxGroup) return;
  const box = new THREE.Box3();
  box.expandByObject(threeCtx.meshGroup);
  box.expandByObject(threeCtx.bboxGroup);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.05);
  const fitDist = (maxDim * 0.56) / Math.tan(THREE.MathUtils.degToRad(threeCtx.camera.fov * 0.5));
  const distance = Math.max(0.24, fitDist);
  const lookDir = new THREE.Vector3(1.0, 0.88, 1.02).normalize();
  if (threeCtx.controls) {
    threeCtx.controls.target.copy(center);
    threeCtx.camera.position.copy(center.clone().addScaledVector(lookDir, distance));
    threeCtx.camera.lookAt(center);
    threeCtx.controls.update();
    return;
  }
  if (threeCtx.fallbackControls) {
    threeCtx.fallbackControls.setTarget(center.x, center.y, center.z);
    threeCtx.fallbackControls.setRadius(distance);
    threeCtx.fallbackControls.update();
    return;
  }
  threeCtx.camera.position.copy(center.clone().addScaledVector(lookDir, distance));
  threeCtx.camera.lookAt(center);
}

function updateMeshScene(payload) {
  initThree();
  if (!threeCtx.scene) return;

  disposeGroup(threeCtx.meshGroup);
  disposeGroup(threeCtx.bboxGroup);

  const meshObj = buildMeshObject(payload?.mesh || {});
  if (meshObj) threeCtx.meshGroup.add(meshObj);
  (payload?.bboxes || []).forEach((row) => {
    const bboxObj = buildBBoxObject(row);
    threeCtx.bboxGroup.add(bboxObj);
  });

  const hasGeometry = Boolean(meshObj) || (payload?.bboxes || []).length > 0;
  const now = Date.now();

  if (
    hasGeometry &&
    !state.meshUserInteracting &&
    now - state.lastMeshInteractionAt >= 10000
  ) {
    focusCameraFromBounds();
  }

  const v = Array.isArray(payload?.mesh?.vertices) ? payload.mesh.vertices.length : 0;
  const t = Array.isArray(payload?.mesh?.triangles) ? payload.mesh.triangles.length : 0;
  const ts = payload?.updated_at || "";
  ui.meshCaption.textContent = `${ts} | vertices=${v} triangles=${t}`;
}

async function loadMesh() {
  const payload = await fetchJson("/api/mesh");
  if (!payload || !payload.mesh) return;
  state.lastMeshPayload = payload;
  updateMeshScene(payload);
}

const TIMING_SERIES = [
  { key: "end_to_end_duration_s", label: "End-to-End", lineClass: "timing-line-e2e" },
  { key: "processing_duration_s", label: "Processing", lineClass: "timing-line-processing" },
  { key: "queue_wait_total_s", label: "Queue Wait", lineClass: "timing-line-queue" },
];

function asDuration(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return null;
  return n;
}

function durationText(value) {
  const n = asDuration(value);
  if (n === null) return "n/a";
  return `${n.toFixed(3)}s`;
}

function timingMetricValue(row, key) {
  const direct = asDuration(row?.[key]);
  if (direct !== null) return direct;

  if (key === "end_to_end_duration_s") {
    return asDuration(row?.total_duration_s);
  }
  if (key === "processing_duration_s") {
    const total = asDuration(row?.total_duration_s);
    const wait = asDuration(row?.vlm_result_queue_wait_s) || 0;
    if (total === null) return null;
    return Math.max(0, total - wait);
  }
  if (key === "queue_wait_total_s") {
    const frameWait = asDuration(row?.frame_queue_wait_s) || 0;
    const stageWait = asDuration(row?.vlm_result_queue_wait_s) || 0;
    const combined = frameWait + stageWait;
    return combined > 0 ? combined : null;
  }
  return null;
}

function timingStats(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const p95 = sorted[Math.max(0, Math.floor((sorted.length - 1) * 0.95))] || 0;
  const avg = sorted.reduce((sum, v) => sum + v, 0) / sorted.length;
  return { avg, p95, latest: values[values.length - 1] };
}

function renderFrameTimingSvg(rows) {
  const sourceRows = rows.slice(-140).map((row, idx) => {
    const metrics = {};
    TIMING_SERIES.forEach((series) => {
      metrics[series.key] = timingMetricValue(row, series.key);
    });
    return {
      idx,
      ts: row?.frame_timestamp || "",
      metrics,
    };
  });

  const visibleRows = sourceRows.filter((row) =>
    TIMING_SERIES.some((series) => row.metrics[series.key] !== null),
  );
  if (visibleRows.length === 0) {
    return `<p class="caption">No timing data available.</p>`;
  }

  const allValues = [];
  visibleRows.forEach((row) => {
    TIMING_SERIES.forEach((series) => {
      const value = row.metrics[series.key];
      if (value !== null) allValues.push(value);
    });
  });
  const sortedAll = allValues.slice().sort((a, b) => a - b);
  const p95All = sortedAll[Math.max(0, Math.floor((sortedAll.length - 1) * 0.95))] || 1e-6;
  const maxV = Math.max(p95All, 1e-6);
  const minV = 0;

  const w = Math.max(720, visibleRows.length * 10);
  const h = 206;
  const pad = 26;
  const usableW = w - pad * 2;
  const usableH = h - pad * 2;

  const pointRows = visibleRows.map((row, i) => {
    const x = pad + (visibleRows.length <= 1 ? usableW * 0.5 : (i / (visibleRows.length - 1)) * usableW);
    const yByKey = {};
    TIMING_SERIES.forEach((series) => {
      const value = row.metrics[series.key];
      if (value === null) return;
      const ratio = Math.min(1.0, (value - minV) / Math.max(maxV - minV, 1e-6));
      yByKey[series.key] = h - pad - ratio * usableH;
    });
    return {
      ts: row.ts,
      x,
      yByKey,
      metrics: row.metrics,
    };
  });

  const polylines = TIMING_SERIES.map((series) => {
    const points = pointRows
      .filter((row) => Number.isFinite(row.yByKey[series.key]))
      .map((row) => `${row.x.toFixed(1)},${row.yByKey[series.key].toFixed(1)}`)
      .join(" ");
    if (!points) return "";
    return `<polyline points="${points}" class="timing-line ${series.lineClass}"></polyline>`;
  }).join("");

  const anchorSeriesKey = "end_to_end_duration_s";
  const circles = pointRows
    .map((row) => {
      const anchorY = row.yByKey[anchorSeriesKey];
      if (!Number.isFinite(anchorY)) return "";
      const title = [
        row.ts || "(unknown timestamp)",
        `end_to_end=${durationText(row.metrics.end_to_end_duration_s)}`,
        `processing=${durationText(row.metrics.processing_duration_s)}`,
        `queue=${durationText(row.metrics.queue_wait_total_s)}`,
      ].join(" | ");
      return `
        <circle
          class="timing-point"
          data-frame-ts="${encodeURIComponent(row.ts || "")}"
          cx="${row.x.toFixed(1)}"
          cy="${anchorY.toFixed(1)}"
          r="3"
        >
          <title>${escapeHtml(title)}</title>
        </circle>
      `;
    })
    .join("");

  const legendRows = TIMING_SERIES.map((series) => {
    const values = pointRows
      .map((row) => row.metrics[series.key])
      .filter((value) => value !== null);
    const stats = timingStats(values);
    const statsText = stats
      ? `latest ${stats.latest.toFixed(3)}s | avg ${stats.avg.toFixed(3)}s | p95 ${stats.p95.toFixed(3)}s`
      : "n/a";
    return `
      <div class="timing-legend-item">
        <span class="swatch ${series.lineClass}"></span>
        <strong>${escapeHtml(series.label)}</strong>
        <span class="tiny">${escapeHtml(statsText)}</span>
      </div>
    `;
  }).join("");

  return `
    <div class="timing-scroll">
      <svg class="timing-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMinYMin meet">
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" class="axis"></line>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" class="axis"></line>
        <line x1="${pad}" y1="${pad}" x2="${w - pad}" y2="${pad}" class="axis axis-guide"></line>
        ${polylines}
        ${circles}
        <text x="${pad}" y="${pad - 8}" class="svg-label">scale p95 ${maxV.toFixed(3)}s</text>
      </svg>
    </div>
    <div class="timing-legend">${legendRows}</div>
    <p class="caption">Tip: click a point to jump to that frame log card.</p>
  `;
}

function highlightFrameCard(frameTs) {
  if (!frameTs) return;
  ui.logsArea.querySelectorAll(".frame-log.is-focused").forEach((node) => {
    node.classList.remove("is-focused");
  });
  const escaped = (window.CSS && CSS.escape) ? CSS.escape(frameTs) : frameTs.replaceAll('"', '\\"');
  const target = ui.logsArea.querySelector(`article.frame-log[data-frame-ts="${escaped}"]`);
  if (!target) return;
  target.classList.add("is-focused");
  target.scrollIntoView({ behavior: "smooth", block: "center" });
}

function bindTimingPointInteractions(container) {
  container.querySelectorAll(".timing-point").forEach((node) => {
    node.addEventListener("click", () => {
      const encoded = node.getAttribute("data-frame-ts") || "";
      const frameTs = decodeURIComponent(encoded);
      highlightFrameCard(frameTs);
    });
  });
}

function stageBars(row) {
  const stages = Array.isArray(row?.stage_timings_ordered) ? row.stage_timings_ordered : [];
  const timed = stages
    .map((s) => ({
      name: s.name || "",
      duration: Number(s.duration_s ?? 0),
      start: s.start_wall || "",
      end: s.end_wall || "",
    }))
    .filter((s) => Number.isFinite(s.duration) && s.duration > 0);

  if (timed.length === 0) {
    return `<p class="caption">No stage timing.</p>`;
  }

  const maxV = Math.max(...timed.map((s) => s.duration), 1e-6);
  return timed
    .map((s) => {
      const width = Math.max(1, Math.round((s.duration / maxV) * 100));
      return `
        <div class="stage-row">
          <div class="stage-meta">${escapeHtml(s.name)} <span>${s.duration.toFixed(3)}s</span></div>
          <div class="stage-bar"><div class="fill" style="width:${width}%"></div></div>
        </div>
      `;
    })
    .join("");
}

function formatNoteRows(notes) {
  if (!notes || typeof notes !== "object") return "";
  const items = Object.entries(notes);
  if (items.length === 0) return "";
  return items
    .map(([k, v]) => {
      const text = typeof v === "string" ? v : JSON.stringify(v);
      return `<div class="note-row"><strong>${escapeHtml(k)}:</strong> ${escapeHtml(text)}</div>`;
    })
    .join("");
}

function createFrameLogCard(row) {
  const node = document.createElement("article");
  node.className = "log-entry frame-log";
  const ts = row.frame_timestamp || "";
  node.dataset.frameTs = ts;
  const world = row.world_name || "";
  const ref = row.reference_found ? "reference found" : "reference missing";
  const total = durationText(row.total_duration_s);
  const endToEnd = durationText(timingMetricValue(row, "end_to_end_duration_s"));
  const processing = durationText(timingMetricValue(row, "processing_duration_s"));
  const queueWait = durationText(timingMetricValue(row, "queue_wait_total_s"));
  const prePipeline = durationText(row.pre_pipeline_duration_s);
  const desc = row.final_description || "";
  const kept = Array.isArray(row.kept_changes) ? row.kept_changes.length : 0;
  const actions = Array.isArray(row.memory_actions) ? row.memory_actions.length : 0;
  const errCount = Array.isArray(row.errors) ? row.errors.length : 0;
  node.innerHTML = `
    <div class="frame-head">
      <div><strong>${escapeHtml(ts)}</strong> <span class="tiny">${escapeHtml(world)}</span></div>
      <div class="tiny">pipeline=${escapeHtml(total)} | e2e=${escapeHtml(endToEnd)} | ${escapeHtml(ref)}</div>
    </div>
    <div class="frame-summary">${escapeHtml(desc || "(none)")}</div>
    <div class="frame-stats tiny">pre_pipeline=${escapeHtml(prePipeline)} queue_wait=${escapeHtml(queueWait)} processing=${escapeHtml(processing)}</div>
    <div class="frame-stats tiny">kept_changes=${kept} memory_actions=${actions} errors=${errCount}</div>
    <div class="stage-wrap">${stageBars(row)}</div>
    <div class="notes-wrap">${formatNoteRows(row.notes)}</div>
    <details>
      <summary>Structured Detail</summary>
      <pre>${escapeHtml(JSON.stringify({
        timing: {
          pre_pipeline_duration_s: row.pre_pipeline_duration_s ?? null,
          frame_queue_wait_s: row.frame_queue_wait_s ?? null,
          vlm_result_queue_wait_s: row.vlm_result_queue_wait_s ?? null,
          queue_wait_total_s: row.queue_wait_total_s ?? null,
          processing_duration_s: row.processing_duration_s ?? null,
          total_duration_s: row.total_duration_s ?? null,
          end_to_end_duration_s: row.end_to_end_duration_s ?? null,
        },
        kept_changes: row.kept_changes || [],
        memory_actions: row.memory_actions || [],
        filter_trace: row.filter_trace || [],
        errors: row.errors || [],
      }, null, 2))}</pre>
    </details>
  `;
  return node;
}

function createSpeechLogCard(row) {
  const node = document.createElement("article");
  node.className = "log-entry";
  const fields = row.fields && typeof row.fields === "object" ? row.fields : {};
  const metrics = Object.entries(fields)
    .filter(([k]) => k !== "text")
    .map(([k, v]) => `<span class="metric">${escapeHtml(k)}=${escapeHtml(v)}</span>`)
    .join(" ");
  const text = fields.text || row.message || "";
  node.innerHTML = `
    <div class="log-summary">${escapeHtml(row.timestamp || "")} ${escapeHtml(row.tag || "")}</div>
    <div class="tiny">${metrics}</div>
    <div class="speech-text">${escapeHtml(text)}</div>
  `;
  return node;
}

function createAppLogCard(row) {
  const node = document.createElement("article");
  node.className = "log-entry";
  node.innerHTML = `
    <div class="log-summary">${escapeHtml(row.timestamp || "")} ${escapeHtml(row.level || "")} ${escapeHtml(row.logger || "")}</div>
    <div>${escapeHtml(row.message || "")}</div>
  `;
  return node;
}

async function loadLogs(kind) {
  const rows = await fetchJson(`/api/logs?kind=${encodeURIComponent(kind)}&limit=300`);
  if (!Array.isArray(rows)) return;
  ui.logsArea.innerHTML = "";
  if (rows.length === 0) {
    ui.logsArea.innerHTML = `<p class="caption">No ${kind} logs yet.</p>`;
    return;
  }

  if (kind === "frame") {
    const chartBlock = document.createElement("div");
    chartBlock.className = "timing-chart";
    chartBlock.innerHTML = `
      <div class="timing-head">
        <strong>Frame Timing Trend</strong>
        <span class="tiny">latest ${rows.length} records</span>
      </div>
      ${renderFrameTimingSvg(rows)}
    `;
    ui.logsArea.appendChild(chartBlock);
    rows.slice().reverse().forEach((row) => ui.logsArea.appendChild(createFrameLogCard(row)));
    bindTimingPointInteractions(chartBlock);
    return;
  }

  if (kind === "speech") {
    rows.slice().reverse().forEach((row) => ui.logsArea.appendChild(createSpeechLogCard(row)));
    return;
  }

  rows.slice().reverse().forEach((row) => ui.logsArea.appendChild(createAppLogCard(row)));
}

async function loadSpeechTimeline() {
  const rows = await fetchJson("/api/speech-timeline?limit=180");
  if (!Array.isArray(rows)) return;
  ui.speechTimeline.innerHTML = "";
  if (rows.length === 0) {
    ui.speechTimeline.innerHTML = `<p class="caption">No spoken output yet.</p>`;
    return;
  }
  rows.slice().reverse().forEach((row) => {
    const node = document.createElement("article");
    node.className = "speech-item";
    const ts = escapeHtml(row.timestamp || "");
    const source = escapeHtml(row.source || "unknown");
    const text = escapeHtml(row.text || "");
    node.innerHTML = `
      <div class="speech-meta">${ts} | source: ${source}</div>
      <div class="speech-text">${text}</div>
    `;
    ui.speechTimeline.appendChild(node);
  });
}

function selectLogTab(kind) {
  state.activeLogKind = kind;
  ui.logTabButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.logKind === kind);
  });
  loadLogs(kind);
}

async function refresh() {
  const meta = await fetchJson("/api/meta");
  if (!meta || !meta.counts || !meta.versions) {
    ui.statusLine.textContent = "Server unavailable. Retrying...";
    return;
  }
  ui.statusLine.textContent = `Connected. Server time: ${meta.server_time || ""}`;

  const previousCounts = state.counts;
  const previousVersions = state.versions;
  state.counts = meta.counts;
  updateChipCounts(state.counts);
  updateSliders();

  if (!previousVersions) {
    if ((state.counts.realtime || 0) > 0) state.realtimeIndex = (state.counts.realtime || 1) - 1;
    if ((state.counts.change || 0) > 0) state.changeIndex = (state.counts.change || 1) - 1;
    updateSliders();
    await loadRealtime(state.realtimeIndex);
    await loadChange(state.changeIndex);
    await loadMemorySummary();
    await loadSpeechTimeline();
    await loadMesh();
    await loadLogs(state.activeLogKind);
    state.versions = meta.versions;
    return;
  }

  const followRealtime = state.realtimeIndex >= Math.max(0, (previousCounts.realtime || 1) - 1);
  const followChange = state.changeIndex >= Math.max(0, (previousCounts.change || 1) - 1);

  if (meta.versions.realtime !== previousVersions.realtime) {
    if (followRealtime) state.realtimeIndex = Math.max(0, (state.counts.realtime || 1) - 1);
    await loadRealtime(state.realtimeIndex);
  }
  if (meta.versions.change !== previousVersions.change) {
    if (followChange) state.changeIndex = Math.max(0, (state.counts.change || 1) - 1);
    await loadChange(state.changeIndex);
  }
  if (meta.versions.change_memory !== previousVersions.change_memory) {
    await loadMemorySummary();
  }
  if (meta.versions.speech_timeline !== previousVersions.speech_timeline) {
    await loadSpeechTimeline();
  }
  if (meta.versions.mesh !== previousVersions.mesh) {
    await loadMesh();
  }

  const key = `${state.activeLogKind}_log`;
  if (meta.versions[key] !== previousVersions[key]) {
    await loadLogs(state.activeLogKind);
  }

  state.versions = meta.versions;
}

function bindEvents() {
  ui.rtSlider.addEventListener("input", async (event) => {
    state.realtimeIndex = Number(event.target.value || 0);
    await loadRealtime(state.realtimeIndex);
  });
  ui.rtPrev.addEventListener("click", async () => {
    state.realtimeIndex = Math.max(0, state.realtimeIndex - 1);
    await loadRealtime(state.realtimeIndex);
  });
  ui.rtNext.addEventListener("click", async () => {
    state.realtimeIndex = Math.min(Math.max(0, state.counts.realtime - 1), state.realtimeIndex + 1);
    await loadRealtime(state.realtimeIndex);
  });

  ui.chSlider.addEventListener("input", async (event) => {
    state.changeIndex = Number(event.target.value || 0);
    await loadChange(state.changeIndex);
  });
  ui.chPrev.addEventListener("click", async () => {
    state.changeIndex = Math.max(0, state.changeIndex - 1);
    await loadChange(state.changeIndex);
  });
  ui.chNext.addEventListener("click", async () => {
    state.changeIndex = Math.min(Math.max(0, state.counts.change - 1), state.changeIndex + 1);
    await loadChange(state.changeIndex);
  });

  ui.logTabButtons.forEach((btn) => {
    btn.addEventListener("click", () => selectLogTab(btn.dataset.logKind || "app"));
  });
}

async function bootstrap() {
  initThree();
  bindEvents();
  await refresh();
  setInterval(refresh, 900);
}

bootstrap();
