document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Metrics
  const statTotalTxns = document.getElementById("stat-total-txns");
  const statApprovedRate = document.getElementById("stat-approved-rate");
  const statFlaggedPct = document.getElementById("stat-flagged-pct");
  const statFlaggedCount = document.getElementById("stat-flagged-count");
  const statAvgRisk = document.getElementById("stat-avg-risk");
  const statPrecision = document.getElementById("stat-precision");
  const statRecall = document.getElementById("stat-recall");
  const statF1 = document.getElementById("stat-f1");
  const statFnrReduction = document.getElementById("stat-fnr-reduction");

  // Metrics Drawer Toggle
  const btnToggleMetrics = document.getElementById("btn-toggle-metrics");
  const btnCloseMetrics = document.getElementById("btn-close-metrics");
  const modelMetricsPanel = document.getElementById("model-metrics-panel");

  // Form & Simulator
  const presetSelector = document.getElementById("preset-selector");
  const simulatorForm = document.getElementById("simulator-form");
  const inputCustomer = document.getElementById("input-customer");
  const inputAmount = document.getElementById("input-amount");
  const inputMerchant = document.getElementById("input-merchant");
  const inputDeviceType = document.getElementById("input-device-type");
  const inputLocation = document.getElementById("input-location");

  const inputUser = document.getElementById("input-user");
  const inputDevice = document.getElementById("input-device");
  const inputTimestamp = document.getElementById("input-timestamp");

  // Simulator Result Box
  const simResultBox = document.getElementById("sim-result-box");
  const resDecisionBadge = document.getElementById("res-decision-badge");
  const resRiskPct = document.getElementById("res-risk-pct");
  const resFactorsList = document.getElementById("res-factors-list");

  // Table
  const filterRisk = document.getElementById("filter-risk");
  const txnsTableBody = document.getElementById("txns-table-body");

  // Modal Sheet
  const detailModal = document.getElementById("detail-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const modalTitle = document.getElementById("modal-title");
  const modalTimestamp = document.getElementById("modal-timestamp");
  const modalDecision = document.getElementById("modal-decision");
  const modalRiskPct = document.getElementById("modal-risk-pct");
  const modalAmount = document.getElementById("modal-amount");
  const modalUserId = document.getElementById("modal-user-id");
  const modalUserName = document.getElementById("modal-user-name");
  const modalHomeCity = document.getElementById("modal-home-city");
  const modalLocation = document.getElementById("modal-location");
  const modalMerchant = document.getElementById("modal-merchant");
  const modalDevice = document.getElementById("modal-device");
  const modalFactorsList = document.getElementById("modal-factors-list");

  // Local Caches
  let customersCache = [];
  let merchantsCache = [];
  let presetsCache = [];

  // Toggle model metrics drawer
  if (btnToggleMetrics) {
    btnToggleMetrics.addEventListener("click", () => {
      modelMetricsPanel.classList.toggle("hidden");
    });
  }
  if (btnCloseMetrics) {
    btnCloseMetrics.addEventListener("click", () => {
      modelMetricsPanel.classList.add("hidden");
    });
  }

  // Close modal handlers
  btnCloseModal.addEventListener("click", () => {
    detailModal.classList.add("hidden");
  });
  detailModal.addEventListener("click", (e) => {
    if (e.target === detailModal) detailModal.classList.add("hidden");
  });

  // Filter change
  filterRisk.addEventListener("change", () => {
    fetchTransactions(filterRisk.value);
  });

  // Customer dropdown change
  inputCustomer.addEventListener("change", () => {
    const custId = inputCustomer.value;
    const customer = customersCache.find(c => c.id === custId);
    if (customer) {
      inputUser.value = customer.id;
      inputLocation.value = customer.home_city;
      updateDeviceValue();
    }
  });

  // Device type change
  inputDeviceType.addEventListener("change", () => {
    updateDeviceValue();
  });

  function updateDeviceValue() {
    const custId = inputCustomer.value;
    const customer = customersCache.find(c => c.id === custId);
    if (inputDeviceType.value === "trusted") {
      inputDevice.value = customer ? customer.primary_device : "DEV_USR_0001_01";
    } else {
      inputDevice.value = "DEV_UNKNOWN_9999";
    }
  }

  // Preset Scenario selector change
  presetSelector.addEventListener("change", (e) => {
    const selectedId = e.target.value;
    const scenario = presetsCache.find(p => p.id === selectedId);
    if (scenario && scenario.payload) {
      const p = scenario.payload;
      inputCustomer.value = p.user_id;
      inputUser.value = p.user_id;
      inputAmount.value = p.amount;
      inputLocation.value = p.location;
      inputMerchant.value = p.merchant_id || "";

      if (p.device_id && p.device_id.includes("UNKNOWN")) {
        inputDeviceType.value = "unrecognized";
        inputDevice.value = p.device_id;
      } else {
        inputDeviceType.value = "trusted";
        inputDevice.value = p.device_id || "DEV_USR_0001_01";
      }

      inputTimestamp.value = p.timestamp || "";
    }
  });

  // Simulator Form Submit
  simulatorForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      user_id: inputUser.value.trim() || inputCustomer.value,
      amount: parseFloat(inputAmount.value),
      location: inputLocation.value.trim(),
      device_id: inputDevice.value.trim(),
      merchant_id: inputMerchant.value.trim() || null,
      timestamp: inputTimestamp.value.trim() || null
    };

    try {
      const resp = await fetch("/api/score-transaction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) throw new Error("Could not score payment");
      const result = await resp.json();

      renderSimulationResult(result);
      fetchStats();
      fetchTransactions(filterRisk.value);
    } catch (err) {
      alert("Error checking payment: " + err.message);
    }
  });

  // Load Dropdowns & Data
  async function loadInitialData() {
    await Promise.all([
      fetchCustomers(),
      fetchMerchants(),
      fetchPresets(),
      fetchStats(),
      fetchTransactions()
    ]);
  }

  async function fetchCustomers() {
    try {
      const resp = await fetch("/api/customers");
      if (!resp.ok) return;
      customersCache = await resp.json();

      inputCustomer.innerHTML = "";
      customersCache.forEach(c => {
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = `${c.name} (${c.home_city})`;
        inputCustomer.appendChild(opt);
      });

      if (customersCache.length > 0) {
        inputCustomer.value = customersCache[0].id;
        inputUser.value = customersCache[0].id;
        inputLocation.value = customersCache[0].home_city;
        updateDeviceValue();
      }
    } catch (err) {
      console.error("Error loading customers:", err);
    }
  }

  async function fetchMerchants() {
    try {
      const resp = await fetch("/api/merchants");
      if (!resp.ok) return;
      merchantsCache = await resp.json();

      inputMerchant.innerHTML = '<option value="">Direct / Contact Transfer</option>';
      merchantsCache.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = `${m.name} (${m.category})`;
        inputMerchant.appendChild(opt);
      });
    } catch (err) {
      console.error("Error loading merchants:", err);
    }
  }

  async function fetchPresets() {
    try {
      const resp = await fetch("/api/sample-scenarios");
      if (!resp.ok) return;
      presetsCache = await resp.json();

      presetSelector.innerHTML = '<option value="">-- Choose a realistic scenario --</option>';
      presetsCache.forEach(sc => {
        const opt = document.createElement("option");
        opt.value = sc.id;
        opt.textContent = `${sc.title} — [${sc.type}]`;
        presetSelector.appendChild(opt);
      });
    } catch (err) {
      console.error("Error loading presets:", err);
    }
  }

  async function fetchStats() {
    try {
      const resp = await fetch("/api/stats");
      if (!resp.ok) return;
      const data = await resp.json();

      const total = data.total_transactions || 0;
      const flagged = data.flagged_count || 0;
      const approvedCount = Math.max(0, total - flagged);
      const approvedRate = total > 0 ? ((approvedCount / total) * 100).toFixed(1) : "100.0";

      statTotalTxns.textContent = total.toLocaleString();
      statApprovedRate.textContent = `${approvedRate}%`;
      statFlaggedPct.textContent = `${data.flagged_pct}%`;
      statFlaggedCount.textContent = `${flagged.toLocaleString()} payments flagged`;
      statAvgRisk.textContent = `${(data.avg_risk_score * 100).toFixed(1)}%`;

      if (data.model_metrics && data.model_metrics.finshield_ensemble) {
        const ensemble = data.model_metrics.finshield_ensemble;
        statPrecision.textContent = `${(ensemble.precision * 100).toFixed(0)}%`;
        statRecall.textContent = `${(ensemble.recall * 100).toFixed(0)}%`;
        statF1.textContent = ensemble.f1_score.toFixed(2);
      }
      if (data.model_metrics && data.model_metrics.improvements) {
        statFnrReduction.textContent = `${data.model_metrics.improvements.fnr_reduction_pct}% False Negative Reduction`;
      }
    } catch (err) {
      console.error("Error fetching stats:", err);
    }
  }

  async function fetchTransactions(filterDecision = "ALL") {
    try {
      let url = "/api/transactions?limit=50";
      if (filterDecision && filterDecision !== "ALL") {
        url += `&risk_level=${filterDecision}`;
      }
      const resp = await fetch(url);
      if (!resp.ok) return;
      const txns = await resp.json();

      if (txns.length === 0) {
        txnsTableBody.innerHTML = `<tr><td colspan="8" class="text-center py-6">No payments matching filter.</td></tr>`;
        return;
      }

      txnsTableBody.innerHTML = "";
      txns.forEach(t => {
        const tr = document.createElement("tr");
        const riskPct = (t.risk_score * 100).toFixed(1);
        const customerName = t.user_name || "Customer";
        const merchantName = t.merchant_name || "Direct Transfer";

        tr.innerHTML = `
          <td>
            <div class="customer-cell">
              <span class="customer-name">${customerName}</span>
              <span class="customer-sub">${t.user_id}</span>
            </div>
          </td>
          <td><strong>₹${t.amount.toLocaleString(undefined, {minimumFractionDigits: 2})}</strong></td>
          <td>${merchantName}</td>
          <td>${t.location}</td>
          <td><span class="status-pill ${t.decision}">${formatDecision(t.decision)}</span></td>
          <td>${riskPct}%</td>
          <td><span class="text-muted">${formatTime(t.timestamp)}</span></td>
          <td><button class="btn-view" data-id="${t.id}">Details</button></td>
        `;

        tr.addEventListener("click", () => {
          openDetailModal(t.id);
        });

        txnsTableBody.appendChild(tr);
      });
    } catch (err) {
      console.error("Error fetching transactions:", err);
    }
  }

  async function openDetailModal(txnId) {
    try {
      const resp = await fetch(`/api/transactions/${txnId}`);
      if (!resp.ok) return;
      const data = await resp.json();

      modalTitle.textContent = `Payment #${data.id}`;
      modalTimestamp.textContent = data.timestamp;
      modalDecision.textContent = formatDecision(data.decision);
      modalDecision.className = `status-pill ${data.decision}`;
      modalRiskPct.textContent = `${(data.risk_score * 100).toFixed(1)}% (${getRiskLabel(data.risk_score)})`;
      modalAmount.textContent = `₹${data.amount.toLocaleString(undefined, {minimumFractionDigits: 2})}`;

      modalUserId.textContent = data.user_id;
      modalUserName.textContent = data.user_name || "Unknown Customer";
      modalHomeCity.textContent = data.home_city || "Unknown";
      modalLocation.textContent = data.location;
      modalMerchant.textContent = data.merchant_name || "Direct Transfer";
      
      const isTrusted = data.device_id && !data.device_id.includes("UNKNOWN");
      modalDevice.textContent = isTrusted ? "Registered Device" : "Unrecognized Device (Alert)";

      renderFactorsList(data.top_factors, modalFactorsList);

      detailModal.classList.remove("hidden");
    } catch (err) {
      console.error("Error opening detail modal:", err);
    }
  }

  function renderSimulationResult(res) {
    simResultBox.classList.remove("hidden");
    resDecisionBadge.textContent = formatDecision(res.decision);
    resDecisionBadge.className = `decision-badge ${res.decision}`;
    resRiskPct.textContent = `${(res.risk_score * 100).toFixed(1)}% (${getRiskLabel(res.risk_score)})`;

    renderFactorsList(res.top_factors, resFactorsList);
  }

  function renderFactorsList(factors, containerElem) {
    containerElem.innerHTML = "";
    if (!factors || factors.length === 0) {
      containerElem.innerHTML = `<span class="text-muted">Standard behavior; no risk anomalies detected.</span>`;
      return;
    }

    factors.forEach(f => {
      const pct = Math.round(f.contribution * 100);
      const friendlyName = humanizeFactor(f.factor);
      const div = document.createElement("div");
      div.className = "factor-item";
      div.innerHTML = `
        <div class="factor-info">
          <span class="factor-name">${friendlyName}</span>
          <span class="factor-pct">+${pct}% risk</span>
        </div>
        <div class="factor-bar-bg">
          <div class="factor-bar-fill" style="width: ${Math.min(pct, 100)}%"></div>
        </div>
      `;
      containerElem.appendChild(div);
    });
  }

  function formatDecision(decision) {
    if (decision === "ALLOW") return "Approved";
    if (decision === "STEP_UP_AUTH") return "Needs OTP";
    if (decision === "BLOCK") return "Declined";
    return decision;
  }

  function getRiskLabel(score) {
    if (score < 0.3) return "Low Risk";
    if (score < 0.7) return "Medium Risk";
    return "High Risk";
  }

  function humanizeFactor(factor) {
    const mapping = {
      "amount_z_score": "Unusually high amount for this customer",
      "Amount deviation": "Unusually high payment amount",
      "is_new_device": "Payment made from an unrecognized device",
      "Unrecognized device": "Payment made from an unrecognized device",
      "location_deviation": "Payment location outside normal home area",
      "Location anomaly": "Payment location outside normal home area",
      "hour_deviation": "Transaction during atypical off-hours",
      "is_new_merchant": "First-time payment to this merchant",
      "velocity_1m": "Multiple rapid payments within 1 minute",
      "velocity_5m": "Sudden burst of payments within 5 minutes",
      "5-minute velocity burst": "Sudden burst of payments within 5 minutes"
    };
    return mapping[factor] || factor;
  }

  function formatTime(timestamp) {
    if (!timestamp) return "";
    const parts = timestamp.split(" ");
    return parts.length > 1 ? parts[1] : timestamp;
  }

  // Run on start
  loadInitialData();

  // Periodic poll
  setInterval(fetchStats, 15000);
});
