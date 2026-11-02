/**
 * Components Module: DOM rendering helpers and UI templates.
 */

export const Components = {
  createClassBadge(classLabel) {
    const isADHD = classLabel.toUpperCase() === "ADHD";
    const cssClass = isADHD ? "adhd" : "control";
    return `<span class="class-pill ${cssClass}">${classLabel}</span>`;
  },

  renderPredictionCard(containerId, result) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const isADHD = result.predicted_class === "ADHD";
    const probPct = (result.adhd_probability * 100).toFixed(1);
    const fillClass = isADHD ? "adhd" : "control";
    const isCorrect = result.is_correct;
    const correctBadge = isCorrect
      ? `<span class="badge" style="background:#dcfce7;color:#15803d;">✓ Correct Match</span>`
      : `<span class="badge" style="background:#fee2e2;color:#b91c1c;">✗ Research Misclassification</span>`;

    container.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div>
          <h3 style="font-size:18px;font-weight:800;color:var(--text-main);">
            Child ${result.participant_id}
          </h3>
          <p style="font-size:12px;color:var(--text-muted);">
            Evaluated using ${result.model_used} across ${result.channels_used.length} channels (${result.montage_used})
          </p>
        </div>
        <div>${correctBadge}</div>
      </div>

      <div class="gauge-wrapper">
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:13px;font-weight:700;">
          <span>Predicted Class: ${Components.createClassBadge(result.predicted_class)}</span>
          <span>True Research Label: ${Components.createClassBadge(result.true_diagnosis)}</span>
        </div>

        <div class="prob-bar-container">
          <div class="prob-bar-fill ${fillClass}" style="width: ${probPct}%"></div>
        </div>

        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-muted);font-weight:600;">
          <span>Control (0%)</span>
          <span style="font-size:14px;font-weight:800;color:var(--text-main);">${probPct}% ADHD Probability</span>
          <span>ADHD (100%)</span>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-top:16px;">
        <div class="biomarker-tile">
          <div class="biomarker-val">${result.confidence_percent}%</div>
          <div class="biomarker-name">Confidence</div>
        </div>
        <div class="biomarker-tile">
          <div class="biomarker-val">${result.epoch_votes.adhd_votes} / ${result.total_epochs}</div>
          <div class="biomarker-name">ADHD Epoch Votes</div>
        </div>
        <div class="biomarker-tile">
          <div class="biomarker-val">${result.epoch_votes.control_votes} / ${result.total_epochs}</div>
          <div class="biomarker-name">Control Epoch Votes</div>
        </div>
        <div class="biomarker-tile">
          <div class="biomarker-val">${result.channels_used.length}</div>
          <div class="biomarker-name">Active Channels</div>
        </div>
      </div>

      <div style="background:#f8fafc;border-left:4px solid #f59e0b;padding:12px 14px;border-radius:4px;margin-top:18px;font-size:12px;color:#78350f;">
        <strong>Scientific Disclaimer:</strong> ${result.disclaimer}
      </div>
    `;
  },

  renderChannelTable(tableBodyId, metricsData) {
    const tbody = document.getElementById(tableBodyId);
    if (!tbody) return;

    const montages = [
      { key: "Full_19ch", name: "Full 19-Channel (Standard 10-20)", n: 19 },
      { key: "Frontal_5ch", name: "Frontal-Central 5-Channel (Fz, Cz, Pz, F3, F4)", n: 5 },
      { key: "Frontal_4ch", name: "Frontal-Central 4-Channel (Fz, Cz, F3, F4)", n: 4 },
      { key: "Midline_2ch", name: "Midline 2-Channel (Fz, Cz - Wearable)", n: 2 }
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
      const reduction = (((19 - m.n) / 19) * 100).toFixed(0) + "%";

      html += `
        <tr>
          <td><strong>${m.name}</strong></td>
          <td><span class="badge" style="background:#e0f2fe;color:#0369a1;">${m.n} channels</span></td>
          <td><span style="color:#dc2626;font-weight:700;">-${reduction}</span></td>
          <td><strong>${svmBalAcc}</strong></td>
          <td>${svmSens}</td>
          <td>${svmSpec}</td>
          <td>${rfBalAcc}</td>
          <td><span class="badge" style="background:#dcfce7;color:#15803d;font-weight:700;">${retention}</span></td>
        </tr>
      `;
    });

    tbody.innerHTML = html;
  }
};
