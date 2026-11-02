/**
 * App Module: Main controller, state management, and event handling.
 */

import { API } from "./api.js";
import { Charts } from "./charts.js";
import { Components } from "./components.js";

// Application State
const State = {
  activeTab: "overview",
  overview: null,
  participants: [],
  selectedParticipant: "v10p",
  selectedChannel: "Cz",
  selectedEpochIdx: 0,
  channelMetrics: null,
  noiseMetrics: null,
  selectedNoiseSnr: 10
};

// DOM Elements
const elements = {
  tabs: document.querySelectorAll(".nav-tab"),
  panes: document.querySelectorAll(".tab-pane"),
  participantSelects: document.querySelectorAll(".participant-select"),
  channelSelect: document.getElementById("signal-channel-select"),
  epochSlider: document.getElementById("signal-epoch-slider"),
  epochLabel: document.getElementById("signal-epoch-label"),
  runPredictBtn: document.getElementById("btn-run-predict"),
  predictModelSelect: document.getElementById("predict-model-select"),
  predictMontageSelect: document.getElementById("predict-montage-select"),
  predictParticipantSelect: document.getElementById("predict-participant-select")
};

// Tab Switching
function initTabs() {
  elements.tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.dataset.tab;
      elements.tabs.forEach(t => t.classList.remove("active"));
      elements.panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
      State.activeTab = targetId;

      // Handle tab-specific activations (refreshing charts for correct dimensions)
      if (targetId === "channels" && State.channelMetrics) {
        Charts.renderChannelComparison("chart-channel-comparison", State.channelMetrics);
      } else if (targetId === "noise" && State.noiseMetrics) {
        Charts.renderNoiseDegradation("chart-noise-degradation", State.noiseMetrics);
      }
    });
  });
}

// Populate Dropdowns
function populateParticipantsDropdown(participants) {
  elements.participantSelects.forEach(select => {
    select.innerHTML = "";
    participants.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.participant_id;
      opt.textContent = `${p.participant_id} (${p.class_label} - ${p.epochs_count} epochs)`;
      select.appendChild(opt);
    });
  });

  if (participants.length > 0) {
    State.selectedParticipant = participants[0].participant_id;
  }
}

// Load and Render Signal Inspector
async function updateSignalInspector() {
  try {
    const trace = await API.getSignalTrace(
      State.selectedParticipant,
      State.selectedChannel,
      State.selectedEpochIdx
    );

    // Update Waveform & PSD Charts
    Charts.renderWaveform("chart-eeg-waveform", trace.time_seconds, trace.voltage_uv, trace.channel);
    Charts.renderPsd("chart-eeg-psd", trace.frequencies_hz, trace.psd_values, trace.channel);

    // Update Biomarker Values
    document.getElementById("val-rel-theta").textContent = trace.band_powers.theta_rel;
    document.getElementById("val-rel-alpha").textContent = trace.band_powers.alpha_rel;
    document.getElementById("val-rel-beta").textContent = trace.band_powers.beta_rel;
    document.getElementById("val-tbr").textContent = trace.band_powers.tbr;
    document.getElementById("val-epoch-info").textContent = `Epoch #${trace.epoch_id} (${trace.class_label})`;

  } catch (err) {
    console.error("Signal trace update error:", err);
  }
}

// Initialize Overview Data
async function loadOverview() {
  try {
    const overview = await API.getOverview();
    State.overview = overview;

    // Populate Overview Stats
    document.getElementById("stat-participants").textContent = overview.participants.total;
    document.getElementById("stat-adhd-ctrl").textContent = `${overview.participants.adhd} ADHD / ${overview.participants.control} Ctrl`;
    document.getElementById("stat-epochs").textContent = overview.epochs.total.toLocaleString();
    document.getElementById("stat-channels").textContent = `${overview.signal.channels_count} Electrodes`;
  } catch (err) {
    console.error("Failed to load overview:", err);
  }
}

// Initialize Channel Comparison Tab
async function loadChannelExperiments() {
  try {
    const data = await API.getChannelExperiments();
    State.channelMetrics = data;
    Charts.renderChannelComparison("chart-channel-comparison", data);
    Components.renderChannelTable("table-channel-comparison-body", data);
  } catch (err) {
    console.error("Failed to load channel experiments:", err);
  }
}

// Initialize Noise Simulation Tab
async function loadNoiseSimulation() {
  try {
    const data = await API.getNoiseSimulation();
    State.noiseMetrics = data;
    Charts.renderNoiseDegradation("chart-noise-degradation", data);
  } catch (err) {
    console.error("Failed to load noise simulation:", err);
  }
}

// Live Prediction Handler
async function handleRunPrediction() {
  const pId = elements.predictParticipantSelect.value;
  const model = elements.predictModelSelect.value;
  const montage = elements.predictMontageSelect.value;

  elements.runPredictBtn.disabled = true;
  elements.runPredictBtn.textContent = "Analyzing Brain Waves...";

  try {
    const result = await API.predictParticipant(pId, model, montage);
    Components.renderPredictionCard("prediction-result-container", result);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    elements.runPredictBtn.disabled = false;
    elements.runPredictBtn.textContent = "🚀 Test Child's Brain Waves";
  }
}

// Bind Events
function bindEvents() {
  // Participant Selector in Inspector
  const signalPSelect = document.getElementById("signal-participant-select");
  if (signalPSelect) {
    signalPSelect.addEventListener("change", (e) => {
      State.selectedParticipant = e.target.value;
      updateSignalInspector();
    });
  }

  // Channel Selector in Inspector
  if (elements.channelSelect) {
    elements.channelSelect.addEventListener("change", (e) => {
      State.selectedChannel = e.target.value;
      updateSignalInspector();
    });
  }

  // Epoch Slider
  if (elements.epochSlider) {
    elements.epochSlider.addEventListener("input", (e) => {
      State.selectedEpochIdx = parseInt(e.target.value, 10);
      if (elements.epochLabel) elements.epochLabel.textContent = `Epoch Index: ${State.selectedEpochIdx}`;
      updateSignalInspector();
    });
  }

  // Live Predict Button
  if (elements.runPredictBtn) {
    elements.runPredictBtn.addEventListener("click", handleRunPrediction);
  }
}

// App Entry Point
async function init() {
  initTabs();
  bindEvents();

  // Load all core data in parallel
  await loadOverview();
  
  try {
    const participants = await API.getParticipants();
    State.participants = participants;
    populateParticipantsDropdown(participants);
  } catch (err) {
    console.error("Failed to load participants:", err);
  }

  // Load initial tab datasets
  updateSignalInspector();
  loadChannelExperiments();
  loadNoiseSimulation();
}

window.addEventListener("DOMContentLoaded", init);

