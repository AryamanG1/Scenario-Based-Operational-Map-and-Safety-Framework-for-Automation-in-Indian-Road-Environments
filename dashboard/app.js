// Module 8: Dynamic Operational Road Map Dashboard
// Illustrative Chandigarh road network with synthetic ODD-readiness scoring.
// All road/scenario data is inlined here (not fetched) so index.html works
// directly over file:// with no local server.

const ROADS = [
  { road_name: "Madhya Marg", road_type: "arterial", coords: [[30.7410, 76.7830], [30.7365, 76.7790], [30.7320, 76.7750]] },
  { road_name: "Dakshin Marg", road_type: "arterial", coords: [[30.7180, 76.7650], [30.7230, 76.7700], [30.7280, 76.7750]] },
  { road_name: "Uttar Marg", road_type: "arterial", coords: [[30.7480, 76.7700], [30.7440, 76.7750], [30.7400, 76.7800]] },
  { road_name: "Jan Marg", road_type: "highway", coords: [[30.7500, 76.7900], [30.7400, 76.7850], [30.7300, 76.7800], [30.7200, 76.7750]] },
  { road_name: "Vikas Marg", road_type: "local", coords: [[30.7350, 76.7950], [30.7300, 76.7900], [30.7250, 76.7850]] },
  { road_name: "Sector 17 Ring Road", road_type: "local", coords: [[30.7410, 76.7830], [30.7420, 76.7870], [30.7390, 76.7900], [30.7360, 76.7870]] },
  { road_name: "Himalaya Marg", road_type: "arterial", coords: [[30.7550, 76.7750], [30.7500, 76.7780], [30.7450, 76.7810]] },
  { road_name: "Sector 22 Link Road", road_type: "local", coords: [[30.7300, 76.7700], [30.7280, 76.7650], [30.7250, 76.7600]] },
  { road_name: "Airport Road", road_type: "highway", coords: [[30.6900, 76.7900], [30.7050, 76.7850], [30.7200, 76.7800]] },
  { road_name: "Zirakpur Road", road_type: "highway", coords: [[30.6500, 76.8200], [30.6800, 76.8000], [30.7100, 76.7900]] },
  { road_name: "Panchkula Road", road_type: "arterial", coords: [[30.7500, 76.8600], [30.7450, 76.8300], [30.7420, 76.8000]] },
  { road_name: "Sector 35 Market Road", road_type: "local", coords: [[30.7250, 76.7550], [30.7280, 76.7580], [30.7310, 76.7610]] },
];

const SCENARIOS = {
  Clear: { visibility: [0.85, 0.95], brightness: [0.80, 0.95], traffic_density: [0.05, 0.15], wetness: [0.0, 0.1] },
  "Heavy Rain": { visibility: [0.20, 0.45], brightness: [0.30, 0.50], traffic_density: [0.30, 0.60], wetness: [0.60, 0.90] },
  "Peak Traffic": { visibility: [0.70, 0.85], brightness: [0.70, 0.85], traffic_density: [0.60, 0.90], wetness: [0.05, 0.15] },
};
const SCENARIO_NAMES = Object.keys(SCENARIOS);

function randInRange(min, max) {
  return min + Math.random() * (max - min);
}

function applyScenarioToRoad(road, scenarioName) {
  const s = SCENARIOS[scenarioName];
  road.scenario = scenarioName;
  road.visibility = randInRange(s.visibility[0], s.visibility[1]);
  road.brightness = randInRange(s.brightness[0], s.brightness[1]);
  road.traffic_density = randInRange(s.traffic_density[0], s.traffic_density[1]);
  road.wetness = randInRange(s.wetness[0], s.wetness[1]);
}

function initRoad(road) {
  applyScenarioToRoad(road, "Clear");
  road.road_quality = randInRange(0.5, 0.9);
  road.pothole_score = randInRange(0.0, 0.15);
  road.detection_confidence = randInRange(0.7, 0.95);
  return road;
}

ROADS.forEach(initRoad);

// ODD score formula, per spec.
function calculateODDScore(road) {
  let score = 100;
  score -= (1 - road.visibility) * 30;
  score -= (1 - road.brightness) * 10;
  score -= road.traffic_density * 25;
  score -= road.wetness * 15;
  score -= road.pothole_score * 10;
  score -= (1 - road.detection_confidence) * 10;
  return Math.max(0, Math.min(100, score));
}

function classifyLevel(score) {
  if (score >= 70) return { level: "L3", color: "#22c55e", label: "Full Automation" };
  if (score >= 40) return { level: "L2", color: "#f59e0b", label: "Supervised" };
  return { level: "L1", color: "#ef4444", label: "Driver Takeover" };
}

const map = L.map("map", { zoomControl: true }).setView([30.7333, 76.7794], 13);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 19,
}).addTo(map);

const polylines = new Map();

function buildPopupContent(road) {
  const score = calculateODDScore(road);
  const { color, label } = classifyLevel(score);
  return `
    <div class="popup-card">
      <div class="popup-header" style="background:${color}">${road.road_name}</div>
      <div class="popup-row"><span>Type</span><span>${road.road_type}</span></div>
      <div class="popup-row"><span>Scenario</span><span>${road.scenario}</span></div>
      <div class="popup-row"><span>ODD Score</span><span>${score.toFixed(1)}</span></div>
      <div class="popup-row"><span>Automation</span><span>${label}</span></div>
      <div class="popup-row"><span>Visibility</span><span>${road.visibility.toFixed(2)}</span></div>
      <div class="popup-row"><span>Brightness</span><span>${road.brightness.toFixed(2)}</span></div>
      <div class="popup-row"><span>Traffic Density</span><span>${road.traffic_density.toFixed(2)}</span></div>
      <div class="popup-row"><span>Wetness</span><span>${road.wetness.toFixed(2)}</span></div>
      <div class="popup-row"><span>Pothole Score</span><span>${road.pothole_score.toFixed(2)}</span></div>
      <div class="popup-row"><span>Detection Conf.</span><span>${road.detection_confidence.toFixed(2)}</span></div>
      <div class="popup-row"><span>Road Quality</span><span>${road.road_quality.toFixed(2)}</span></div>
    </div>
  `;
}

function renderRoads() {
  ROADS.forEach((road) => {
    const score = calculateODDScore(road);
    const { color } = classifyLevel(score);
    const line = L.polyline(road.coords, {
      color,
      weight: 6,
      opacity: 0.85,
      lineCap: "round",
    }).addTo(map);
    line.bindPopup(buildPopupContent(road));
    polylines.set(road.road_name, line);
  });
}

function updateRoadVisual(road) {
  const score = calculateODDScore(road);
  const { color } = classifyLevel(score);
  const line = polylines.get(road.road_name);
  line.setStyle({ color });
  line.setPopupContent(buildPopupContent(road));
}

function updateCounts() {
  let l3 = 0;
  let l2 = 0;
  let l1 = 0;
  ROADS.forEach((road) => {
    const { level } = classifyLevel(calculateODDScore(road));
    if (level === "L3") l3 += 1;
    else if (level === "L2") l2 += 1;
    else l1 += 1;
  });
  document.getElementById("count-l3").textContent = l3;
  document.getElementById("count-l2").textContent = l2;
  document.getElementById("count-l1").textContent = l1;
}

function pickCluster(size) {
  const startIdx = Math.floor(Math.random() * ROADS.length);
  const cluster = [];
  for (let i = 0; i < size; i += 1) {
    cluster.push(ROADS[(startIdx + i) % ROADS.length]);
  }
  return cluster;
}

function injectScenario() {
  const scenarioName = SCENARIO_NAMES[Math.floor(Math.random() * SCENARIO_NAMES.length)];
  const clusterSize = 2 + Math.floor(Math.random() * 3); // 2-4 localized roads
  const cluster = pickCluster(clusterSize);

  cluster.forEach((road) => {
    applyScenarioToRoad(road, scenarioName);
    updateRoadVisual(road);
  });

  document.getElementById("scenario-value").textContent =
    `${scenarioName} — ${cluster.map((r) => r.road_name).join(", ")}`;

  updateCounts();
}

function renderPipelineStats() {
  const container = document.getElementById("pipeline-stats");
  if (typeof PIPELINE_STATS === "undefined") {
    return; // pipeline_stats.js not generated yet -- keep the default note.
  }

  const s = PIPELINE_STATS;
  const modeRow = (mode, cls) =>
    `<div class="count-row"><span class="dot ${cls}"></span> ${mode} <span>${s.final_mode_counts[mode] || 0}</span></div>`;

  container.innerHTML = `
    <p class="pipeline-note">${s.num_scenes} real IDD-Lite scenes processed.</p>
    ${modeRow("Normal", "dot-l3")}
    ${modeRow("Degraded", "dot-l2")}
    ${modeRow("Takeover", "dot-l1")}
    <p class="pipeline-note">Mean ODD score: ${s.mean_odd_score.toFixed(1)} / 100</p>
    <p class="pipeline-note">Automation-feasible: ${(s.fraction_automation_feasible * 100).toFixed(1)}%</p>
  `;
}

// Stage 8: Closed-Loop Simulation replay marker.
//
// closed_loop_runner.py writes one full (finished) simulation run as
// `const CARLA_LIVE = {...}` to carla_live.js (same file:// reasoning as
// PIPELINE_STATS above -- see write_carla_live_js()'s docstring). Since a
// run is a bounded, already-completed tick sequence, this replays it
// client-side with setInterval rather than polling a live server -- no
// new infrastructure, works over plain file:// like the rest of this
// dashboard. If carla_live.js hasn't been generated yet, CARLA_LIVE is
// undefined and this whole block is a silent no-op.
const CARLA_MODE_COLORS = { Normal: "#22c55e", Degraded: "#f59e0b", Takeover: "#ef4444" };

function interpolateAlongPolyline(coords, progress) {
  if (coords.length === 1) return coords[0];
  const clamped = Math.max(0, Math.min(1, progress));
  const segLengths = [];
  let total = 0;
  for (let i = 0; i < coords.length - 1; i += 1) {
    const d = L.latLng(coords[i]).distanceTo(L.latLng(coords[i + 1]));
    segLengths.push(d);
    total += d;
  }
  let target = clamped * total;
  for (let i = 0; i < segLengths.length; i += 1) {
    if (target <= segLengths[i] || i === segLengths.length - 1) {
      const t = segLengths[i] === 0 ? 0 : target / segLengths[i];
      const [lat1, lng1] = coords[i];
      const [lat2, lng2] = coords[i + 1];
      return [lat1 + (lat2 - lat1) * t, lng1 + (lng2 - lng1) * t];
    }
    target -= segLengths[i];
  }
  return coords[coords.length - 1];
}

function renderCarlaLiveStats(tick, meta) {
  const container = document.getElementById("carla-live-stats");
  if (!container) return;
  container.innerHTML = `
    <p class="pipeline-note">${meta.used_carla ? "Live CARLA server" : "Local kinematic fallback (no CARLA attached)"} — replaying ${meta.ticks.length} ticks on ${meta.road_name}.</p>
    <div class="count-row"><span class="dot" style="background:${CARLA_MODE_COLORS[tick.mode] || "#888"}"></span> Mode <span>${tick.mode}</span></div>
    <div class="count-row"><span>ADS State</span><span>${tick.ads_state}</span></div>
    <div class="count-row"><span>RSS Distance</span><span>${tick.rss_distance.toFixed(1)} m</span></div>
    <div class="count-row"><span>Stability Margin</span><span>${tick.stability_margin.toFixed(3)}</span></div>
    <div class="count-row"><span>Ego Speed</span><span>${(tick.v_ego * 3.6).toFixed(1)} km/h</span></div>
  `;
}

function initCarlaReplay() {
  if (typeof CARLA_LIVE === "undefined" || !CARLA_LIVE.ticks || CARLA_LIVE.ticks.length === 0) {
    return;
  }
  const road = ROADS.find((r) => r.road_name === CARLA_LIVE.road_name) || ROADS[0];
  const marker = L.circleMarker(road.coords[0], {
    radius: 9,
    color: "#1d4ed8",
    weight: 2,
    fillColor: "#3b82f6",
    fillOpacity: 0.9,
  }).addTo(map);

  let i = 0;
  const step = () => {
    const tick = CARLA_LIVE.ticks[i % CARLA_LIVE.ticks.length];
    const latlng = interpolateAlongPolyline(road.coords, tick.progress);
    const color = CARLA_MODE_COLORS[tick.mode] || "#3b82f6";
    marker.setLatLng(latlng);
    marker.setStyle({ fillColor: color, color: "#1d4ed8" });
    marker.bindPopup(`
      <div class="popup-card">
        <div class="popup-header" style="background:${color}">Closed-Loop Sim — tick ${tick.tick}</div>
        <div class="popup-row"><span>Mode</span><span>${tick.mode}</span></div>
        <div class="popup-row"><span>ADS State</span><span>${tick.ads_state}</span></div>
        <div class="popup-row"><span>RSS Distance</span><span>${tick.rss_distance.toFixed(1)} m</span></div>
        <div class="popup-row"><span>Stability Margin</span><span>${tick.stability_margin.toFixed(3)}</span></div>
        <div class="popup-row"><span>Ego Speed</span><span>${(tick.v_ego * 3.6).toFixed(1)} km/h</span></div>
      </div>
    `);
    renderCarlaLiveStats(tick, CARLA_LIVE);
    i += 1;
  };

  step();
  setInterval(step, 500);
}

renderRoads();
updateCounts();
renderPipelineStats();
initCarlaReplay();
document.getElementById("scenario-value").textContent = "Clear (baseline)";

setInterval(injectScenario, 5000);
