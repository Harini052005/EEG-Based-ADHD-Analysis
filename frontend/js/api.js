/**
 * API Client Module: Handles asynchronous REST requests to backend endpoints.
 */

const API_BASE = ""; // Relative path to current host (same origin)

export const API = {
  async getOverview() {
    const res = await fetch(`${API_BASE}/api/overview`);
    if (!res.ok) throw new Error("Failed to fetch overview data");
    const json = await res.json();
    return json.data;
  },

  async getParticipants() {
    const res = await fetch(`${API_BASE}/api/participants`);
    if (!res.ok) throw new Error("Failed to fetch participants list");
    const json = await res.json();
    return json.data;
  },

  async getSignalTrace(participantId = null, channel = "Cz", epochIdx = 0) {
    let url = `${API_BASE}/api/signal/trace?channel=${encodeURIComponent(channel)}&epoch_idx=${epochIdx}`;
    if (participantId) {
      url += `&participant_id=${encodeURIComponent(participantId)}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch signal trace");
    const json = await res.json();
    return json.data;
  },

  async getChannelExperiments() {
    const res = await fetch(`${API_BASE}/api/channel-experiments`);
    if (!res.ok) throw new Error("Failed to fetch channel experiments data");
    const json = await res.json();
    return json.data;
  },

  async getNoiseSimulation() {
    const res = await fetch(`${API_BASE}/api/noise-simulation`);
    if (!res.ok) throw new Error("Failed to fetch noise simulation data");
    const json = await res.json();
    return json.data;
  },

  async predictParticipant(participantId, model = "svm", montage = "full_19ch") {
    const res = await fetch(`${API_BASE}/api/predict/participant`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ participant_id: participantId, model, montage })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || "Prediction failed");
    }
    const json = await res.json();
    return json.data;
  },

  async predictEpoch(epochId, model = "svm", montage = "full_19ch") {
    const res = await fetch(`${API_BASE}/api/predict/epoch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epoch_id: epochId, model, montage })
    });
    if (!res.ok) throw new Error("Epoch prediction failed");
    const json = await res.json();
    return json.data;
  }
};
