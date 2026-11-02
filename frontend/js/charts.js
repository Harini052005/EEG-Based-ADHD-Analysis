/**
 * Charts Module: Manages Chart.js interactive visualizations.
 */

// Store chart instances to allow clean updates without overlapping canvases
const chartInstances = {};

function destroyChart(canvasId) {
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
    delete chartInstances[canvasId];
  }
}

export const Charts = {
  renderWaveform(canvasId, timeArr, voltageArr, channelName, color = "#2563eb") {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId).getContext("2d");

    chartInstances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: timeArr.map(t => t.toFixed(2) + "s"),
        datasets: [{
          label: `${channelName} Scalp Voltage (µV)`,
          data: voltageArr,
          borderColor: color,
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: {
            title: { display: true, text: "Time (seconds)", font: { weight: "bold", size: 11 } },
            ticks: { maxTicksLimit: 10 }
          },
          y: {
            title: { display: true, text: "Amplitude (µV)", font: { weight: "bold", size: 11 } }
          }
        },
        plugins: {
          legend: { display: true, position: "top", labels: { boxWidth: 12 } },
          tooltip: {
            mode: "index",
            intersect: false,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} µV`
            }
          }
        }
      }
    });
  },

  renderPsd(canvasId, freqsArr, psdArr, channelName) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId).getContext("2d");

    chartInstances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: freqsArr.map(f => f.toFixed(1) + " Hz"),
        datasets: [{
          label: `${channelName} Power Spectral Density (µV²/Hz)`,
          data: psdArr,
          borderColor: "#0f766e",
          backgroundColor: "rgba(15, 118, 110, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: {
            title: { display: true, text: "Frequency (Hz) [Delta 0.5-4 | Theta 4-8 | Alpha 8-13 | Beta 13-30]", font: { weight: "bold", size: 11 } },
            ticks: { maxTicksLimit: 12 }
          },
          y: {
            title: { display: true, text: "PSD (µV² / Hz)", font: { weight: "bold", size: 11 } },
            beginAtZero: true
          }
        },
        plugins: {
          legend: { display: true, position: "top" }
        }
      }
    });
  },

  renderChannelComparison(canvasId, metricsData) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId).getContext("2d");

    const montages = ["Full_19ch", "Frontal_5ch", "Frontal_4ch", "Midline_2ch"];
    const labels = ["Full 19ch", "Frontal 5ch", "Frontal 4ch", "Midline 2ch"];

    const rfData = montages.map(m => {
      const entry = metricsData[m]?.random_forest?.participant_level?.mean?.balanced_accuracy;
      return entry ? (entry * 100).toFixed(1) : 0;
    });

    const svmData = montages.map(m => {
      const entry = metricsData[m]?.svm_rbf?.participant_level?.mean?.balanced_accuracy;
      return entry ? (entry * 100).toFixed(1) : 0;
    });

    chartInstances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "SVM (RBF Kernel)",
            data: svmData,
            backgroundColor: "#2563eb",
            borderRadius: 6
          },
          {
            label: "Random Forest",
            data: rfData,
            backgroundColor: "#0f766e",
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 40,
            max: 80,
            title: { display: true, text: "Subject Balanced Accuracy (%)", font: { weight: "bold" } }
          }
        },
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              afterBody: (items) => {
                const idx = items[0].dataIndex;
                const mKey = montages[idx];
                const nCh = metricsData[mKey]?.channels?.length || "";
                return `Channel Count: ${nCh} electrodes`;
              }
            }
          }
        }
      }
    });
  },

  renderNoiseDegradation(canvasId, noiseData) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId).getContext("2d");

    const snrKeys = ["Clean", "+20dB", "+10dB", "+5dB", "+0dB", "-5dB"];
    const snrLabels = ["Clean (Inf)", "+20 dB", "+10 dB", "+5 dB", "0 dB", "-5 dB"];

    const full19 = snrKeys.map(k => (noiseData.Full_19ch?.SVM_RBF?.[k]?.part_bacc_mean * 100 || 0).toFixed(1));
    const front5 = snrKeys.map(k => (noiseData.Frontal_5ch?.SVM_RBF?.[k]?.part_bacc_mean * 100 || 0).toFixed(1));
    const mid2   = snrKeys.map(k => (noiseData.Midline_2ch?.SVM_RBF?.[k]?.part_bacc_mean * 100 || 0).toFixed(1));

    chartInstances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: snrLabels,
        datasets: [
          {
            label: "Full 19-Channel (SVM)",
            data: full19,
            borderColor: "#2563eb",
            backgroundColor: "#2563eb",
            borderWidth: 2.5,
            tension: 0.2
          },
          {
            label: "Frontal 5-Channel (SVM)",
            data: front5,
            borderColor: "#0f766e",
            backgroundColor: "#0f766e",
            borderWidth: 2,
            tension: 0.2
          },
          {
            label: "Midline 2-Channel (SVM)",
            data: mid2,
            borderColor: "#dc2626",
            backgroundColor: "#dc2626",
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 40,
            max: 75,
            title: { display: true, text: "Subject Balanced Accuracy (%)", font: { weight: "bold" } }
          },
          x: {
            title: { display: true, text: "Signal-to-Noise Ratio (SNR)", font: { weight: "bold" } }
          }
        },
        plugins: {
          legend: { position: "top" }
        }
      }
    });
  }
};
