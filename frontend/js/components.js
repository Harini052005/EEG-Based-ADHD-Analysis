/**
 * Components Module: DOM rendering helpers with pediatric styling & plain language.
 * Palette: #E11A45 (Rose), #EDC92C (Yellow), #E3AFC9 (Mauve), #EFECD7 (Cream)
 */

export const Components = {
  createClassBadge(classLabel) {
    const isADHD = classLabel.toUpperCase() === "ADHD";
    if (isADHD) {
      return `<span style="background:#FDF0F3;color:#E11A45;border:1.5px solid #E3AFC9;padding:4px 12px;border-radius:9999px;font-weight:800;font-size:12.5px;">🧸 ADHD Group</span>`;
    } else {
      return `<span style="background:#FEF9E6;color:#8E6A1B;border:1.5px solid #EDC92C;padding:4px 12px;border-radius:9999px;font-weight:800;font-size:12.5px;">🌟 Control Group</span>`;
    }
  },

  renderPredictionCard(containerId, result) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const isADHD = result.predicted_class === "ADHD";
    const probPct = (result.adhd_probability * 100).toFixed(1);
    const fillClass = isADHD ? "adhd" : "control";
    const isCorrect = result.is_correct;

    const matchBadge = isCorrect
      ? `<span style="background:#fce8ed;color:#E11A45;border:1.5px solid #E3AFC9;padding:5px 14px;border-radius:9999px;font-weight:800;font-size:13px;">✓ Correct Prediction</span>`
      : `<span style="background:#FEF9E6;color:#8E6A1B;border:1.5px solid #EDC92C;padding:5px 14px;border-radius:9999px;font-weight:800;font-size:13px;">⚠️ Did Not Match Record</span>`;

    container.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
        <div>
          <h3 style="font-size:20px;font-weight:800;color:var(--text-main);">
            🧸 Child ${result.participant_id}
          </h3>
          <p style="font-size:13px;color:var(--text-muted);font-weight:600;">
            Tested with ${result.model_used} using ${result.channels_used.length} head sensors
          </p>
        </div>
        <div>${matchBadge}</div>
      </div>

      <div class="gauge-card">
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:13px;font-weight:700;">
          <span>Computer Verdict: ${Components.createClassBadge(result.predicted_class)}</span>
          <span>Actual Record: ${Components.createClassBadge(result.true_diagnosis)}</span>
        </div>

        <div class="prob-track">
          <div class="prob-fill ${fillClass}" style="width: ${probPct}%"></div>
        </div>

        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-muted);font-weight:700;">
          <span>🌟 Control (0%)</span>
          <span style="font-size:15px;font-weight:800;color:var(--c-primary);">${probPct}% Estimated Probability</span>
          <span>🧸 ADHD (100%)</span>
        </div>
      </div>

      <div class="biomarker-grid" style="margin-top:16px;">
        <div class="biomarker-card highlight">
          <div class="value">${result.confidence_percent}%</div>
          <div class="label">Certainty Level</div>
        </div>
        <div class="biomarker-card">
          <div class="value">${result.epoch_votes.adhd_votes} of ${result.total_epochs}</div>
          <div class="label">Recordings Voted ADHD</div>
        </div>
        <div class="biomarker-card">
          <div class="value">${result.epoch_votes.control_votes} of ${result.total_epochs}</div>
          <div class="label">Recordings Voted Control</div>
        </div>
        <div class="biomarker-card">
          <div class="value">${result.channels_used.length}</div>
          <div class="label">Sensors Read</div>
        </div>
      </div>

      <div style="background:#FEF9E6;border-left:4px solid #EDC92C;padding:12px 16px;border-radius:8px;margin-top:18px;font-size:12.5px;color:#7A5700;">
        <strong>Friendly Note:</strong> ${result.disclaimer}
      </div>
    `;
  },

  renderChannelTable(tableBodyId, metricsData) {
    const tbody = document.getElementById(tableBodyId);
    if (!tbody) return;

    const montages = [
      { key: "Full_19ch", name: "🧢 Full Cap (19 Sensors)", n: 19 },
      { key: "Frontal_5ch", name: "🎧 Headband (5 Sensors: Fz, Cz, Pz, F3, F4)", n: 5 },
      { key: "Frontal_4ch", name: "🎧 Forehead Band (4 Sensors: Fz, Cz, F3, F4)", n: 4 },
      { key: "Midline_2ch", name: "🎈 Simple 2-Sensor Band (Fz & Cz)", n: 2 }
    ];

    const fullSvmBalAcc = metricsData["Full_19ch"]?.svm_rbf?.participant_level?.mean?.balanced_accuracy || 0.6756;

    let html = "";
    montages.forEach(m => {
      const svm = metricsData[m.key]?.svm_rbf?.participant_level?.mean;
      const rf = metricsData[m.key]?.random_forest?.participant_level?.mean;

      const svmBalAcc = svm ? (svm.balanced_accuracy * 100).toFixed(1) + "%" : "N/A";
      const svmSens   = svm ? (svm.sensitivity * 100).toFixed(1) + "%" : "N/A";
      const svmSpec   = svm ? (svm.specificity * 100).toFixed(1) + "%" : "N/A";
      const rfBalAcc  = rf ? (rf.balanced_accuracy * 100).toFixed(1) + "%" : "N/A";

      const retention = svm ? ((svm.balanced_accuracy / fullSvmBalAcc) * 100).toFixed(1) + "%" : "N/A";
      const reduction = (((19 - m.n) / 19) * 100).toFixed(0) + "% fewer wires";

      html += `
        <tr>
          <td><strong>${m.name}</strong></td>
          <td><span style="background:#EFECD7;padding:4px 10px;border-radius:9999px;font-weight:700;">${m.n} electrodes</span></td>
          <td><span style="color:#E11A45;font-weight:800;">${reduction}</span></td>
          <td><strong style="color:#E11A45;font-size:15px;">${svmBalAcc}</strong></td>
          <td>${svmSens}</td>
          <td>${svmSpec}</td>
          <td>${rfBalAcc}</td>
          <td><span style="background:#FDF0F3;color:#E11A45;border:1.5px solid #E3AFC9;padding:4px 10px;border-radius:9999px;font-weight:800;">${retention} Kept</span></td>
        </tr>
      `;
    });

    tbody.innerHTML = html;
  }
};
