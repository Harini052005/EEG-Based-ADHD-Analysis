/**
 * Charts Module: Manages interactive Chart.js visualizations.
 * Follows strict palette:
 * - #E11A45 (Primary Rose-Crimson)
 * - #EDC92C (Honey Yellow)
 * - #E3AFC9 (Rose Mauve)
 * - #EFECD7 (Cream Ivory)
 */

const chartInstances = {};

function destroyChart(canvasId) {
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
    delete chartInstances[canvasId];
  }
}

export const Charts = {
  renderWaveform(canvasId, timeArr, voltageArr, channelName, color = "#E11A45") {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    chartInstances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: timeArr.map(t => t.toFixed(2) + "s"),
        datasets: [{
          label: `Sensor ${channelName} Voltage (µV)`,
          data: voltageArr,
          borderColor: color,
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          backgroundColor: "rgba(225, 26, 69, 0.05)",
          tension: 0.2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Time in Seconds (0.0 to 2.0s)", font: { weight: "bold", size: 12 }, color: "#6C555E" },
            ticks: { maxTicksLimit: 10, color: "#6C555E" },
            grid: { color: "rgba(229, 223, 202, 0.6)" }
          },
          y: {
            title: { display: true, text: "Brain Signal Microvolts (µV)", font: { weight: "bold", size: 12 }, color: "#6C555E" },
            ticks: { color: "#6C555E" },
            grid: { color: "rgba(229, 223, 202, 0.6)" }
          }
        },
        plugins: {
          legend: { display: true, position: "top", labels: { color: "#2A171D", font: { weight: "bold" } } },
          tooltip: {
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
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    chartInstances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: freqsArr.map(f => f.toFixed(1) + " Hz"),
        datasets: [{
          label: `Sensor ${channelName} Brain Waves (Welch PSD)`,
          data: psdArr,
          borderColor: "#E11A45",
          backgroundColor: "rgba(227, 175, 201, 0.25)",
          borderWidth: 2.5,
          pointRadius: 0,
          fill: true,
          tension: 0.25
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Frequency (Hz) [Theta 4-8Hz | Alpha 8-13Hz | Beta 13-30Hz]", font: { weight: "bold", size: 12 }, color: "#6C555E" },
            ticks: { maxTicksLimit: 12, color: "#6C555E" },
            grid: { color: "rgba(229, 223, 202, 0.6)" }
          },
          y: {
            title: { display: true, text: "Power Spectral Density (µV²/Hz)", font: { weight: "bold", size: 12 }, color: "#6C555E" },
            ticks: { color: "#6C555E" },
            grid: { color: "rgba(229, 223, 202, 0.6)" },
            beginAtZero: true
          }
        },
        plugins: {
          legend: { display: true, position: "top", labels: { color: "#2A171D", font: { weight: "bold" } } }
        }
      }
    });
  },

  renderChannelComparison(canvasId, metricsData) {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const montages = ["Full_19ch", "Frontal_5ch", "Frontal_4ch", "Midline_2ch"];
    const labels = ["Full Cap (19 Sensors)", "Headband (5 Sensors)", "Forehead (4 Sensors)", "Simple Band (2 Sensors)"];

    const svmData = montages.map(m => {
      const entry = metricsData[m]?.svm_rbf?.participant_level?.mean?.balanced_accuracy;
      return entry ? (entry * 100).toFixed(1) : 0;
    });

    const rfData = montages.map(m => {
      const entry = metricsData[m]?.random_forest?.participant_level?.mean?.balanced_accuracy;
      return entry ? (entry * 100).toFixed(1) : 0;
    });

    chartInstances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "SVM Model (Primary)",
            data: svmData,
            backgroundColor: "#E11A45",
            borderRadius: 8
          },
          {
            label: "Random Forest Model",
            data: rfData,
            backgroundColor: "#EDC92C",
            borderRadius: 8
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 45,
            max: 75,
            title: { display: true, text: "Child Balanced Accuracy (%)", font: { weight: "bold", size: 12 }, color: "#6C555E" },
            ticks: { color: "#6C555E" },
            grid: { color: "rgba(229, 223, 202, 0.6)" }
          },
          x: {
            ticks: { color: "#2A171D", font: { weight: "bold" } },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { position: "top", labels: { color: "#2A171D", font: { weight: "bold" } } }
        }
      }
    });
  },

  renderNoiseDegradation(canvasId, noiseData) {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const snrKeys = ["Clean", "+20dB", "+10dB", "+5dB", "+0dB", "-5dB"];
    const snrLabels = ["Clean (Zero Noise)", "Mild (+20 dB)", "Normal (+10 dB)", "Wiggly (+5 dB)", "Heavy (0 dB)", "Extreme (-5 dB)"];

    const full19 = snrKeys.map(k => (noiseData.Full_19ch?.SVM_RBF?.[k]?.part_bacc_mean * 100 || 0).toFixed(1));
    const front5 = snrKeys.map(k => (noiseData.Frontal_5ch?.SVM_RBF?.[k]?.part_bacc_mean * 100 || 0).toFixed(1));
    const mid2   = snrKeys.map(k => (noiseData.Midline_2ch?.SVM_RBF?.[k]?.part_bacc_mean * 100 || 0).toFixed(1));

    chartInstances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: snrLabels,
        datasets: [
          {
            label: "🧢 Full 19-Sensor Cap",
            data: full19,
            borderColor: "#E11A45",
            backgroundColor: "#E11A45",
            borderWidth: 3,
            tension: 0.2
          },
          {
            label: "🎧 5-Sensor Headband",
            data: front5,
            borderColor: "#EDC92C",
            backgroundColor: "#EDC92C",
            borderWidth: 2.5,
            tension: 0.2
          },
          {
            label: "🎈 Simple 2-Sensor Band",
            data: mid2,
            borderColor: "#E3AFC9",
            backgroundColor: "#E3AFC9",
            borderWidth: 2.5,
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
            title: { display: true, text: "Child Balanced Accuracy (%)", font: { weight: "bold", size: 12 }, color: "#6C555E" },
            ticks: { color: "#6C555E" },
            grid: { color: "rgba(229, 223, 202, 0.6)" }
          },
          x: {
            title: { display: true, text: "Signal-to-Noise Level (How much noise was added)", font: { weight: "bold", size: 12 }, color: "#6C555E" },
            ticks: { color: "#2A171D", font: { weight: "bold" } },
            grid: { color: "rgba(229, 223, 202, 0.6)" }
          }
        },
        plugins: {
          legend: { position: "top", labels: { color: "#2A171D", font: { weight: "bold" } } }
        }
      }
    });
  }
};
