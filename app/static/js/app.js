/**
 * app.js - Unified Client Controller for ML Capstone Dashboard
 * Covers 3 Tracks:
 * 1. Regression (Diamond Price Prediction)
 * 2. Classification (Fraud Detection)
 * 3. Clustering (Amazon Product Catalog Segmentation)
 */

document.addEventListener('DOMContentLoaded', () => {
  console.log('[ML Capstone] Initializing Application Controller...');

  // =========================================================================
  // GLOBAL STATE
  // =========================================================================
  const State = {
    activeTrack: 'clustering',
    health: null,
    regression: {
      info: null,
      charts: { actualPred: null, residual: null, featImp: null }
    },
    classification: {
      info: null,
      selectedChartModel: 'Random Forest',
      charts: { confusion: null, roc: null, metrics: null }
    },
    clustering: {
      overview: null,
      pcaData: null,
      activeElbowMetric: 'inertia',
      catalog: {
        page: 1,
        pageSize: 12,
        totalPages: 1,
        searchQuery: '',
        category: 'All',
        cluster: 'All',
        sortBy: 'price_desc'
      },
      userPoint: null,
      charts: { size: null, profile: null, pca: null, elbow: null }
    }
  };

  const CLUSTER_COLORS = {
    0: '#3b82f6', // Blue - Evergreen Standard
    1: '#ef4444', // Red - High Velocity
    2: '#10b981', // Emerald - Discount Driven
    3: '#8b5cf6'  // Purple - Cold Start / Dormant
  };

  // Default Chart.js Options for Dark Institutional Theme
  const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600, easing: 'easeOutQuart' },
    plugins: {
      legend: {
        labels: {
          color: '#cbd5e1',
          font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
          boxWidth: 12,
          padding: 12
        }
      },
      tooltip: {
        backgroundColor: '#121927',
        borderColor: '#1e293b',
        borderWidth: 1,
        titleColor: '#f8fafc',
        titleFont: { family: "'Inter', sans-serif", size: 12, weight: '600' },
        bodyColor: '#cbd5e1',
        bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
        padding: 10,
        boxPadding: 4,
        cornerRadius: 6
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)', borderColor: '#1e293b' },
        ticks: { color: '#94a3b8', font: { family: "'Inter', sans-serif", size: 11 } }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)', borderColor: '#1e293b' },
        ticks: { color: '#94a3b8', font: { family: "'Inter', sans-serif", size: 11 } }
      }
    }
  };

  // =========================================================================
  // NAVIGATION & TAB SWITCHING
  // =========================================================================
  function initNavigation() {
    // Track Sidebar Switching
    const trackNavItems = document.querySelectorAll('.sidebar-nav .nav-item');
    const trackPanes = document.querySelectorAll('.track-pane');

    trackNavItems.forEach(btn => {
      btn.addEventListener('click', () => {
        const track = btn.dataset.track;
        if (!track || State.activeTrack === track) return;

        trackNavItems.forEach(b => {
          b.classList.remove('active');
          b.setAttribute('aria-selected', 'false');
        });
        trackPanes.forEach(pane => {
          pane.classList.remove('active');
          pane.setAttribute('aria-hidden', 'true');
        });

        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
        const activePane = document.getElementById(`track-${track}`);
        if (activePane) {
          activePane.classList.add('active');
          activePane.setAttribute('aria-hidden', 'false');
        }
        State.activeTrack = track;

        // Resize charts on reveal
        resizeActiveTrackCharts(track);
      });
    });

    // Sub-panel Tabs Switching within tracks
    const trackTabs = document.querySelectorAll('.track-tab');
    trackTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const parentHeader = tab.closest('.track-pane');
        if (!parentHeader) return;

        const siblings = parentHeader.querySelectorAll('.track-tab');
        const subPanels = parentHeader.querySelectorAll('.sub-panel');
        const targetId = tab.dataset.target;

        siblings.forEach(t => t.classList.remove('active'));
        subPanels.forEach(p => p.classList.remove('active'));

        tab.classList.add('active');
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
          targetPanel.classList.add('active');
        }

        // Trigger chart resize if needed
        resizeActiveTrackCharts(State.activeTrack);
      });
    });
  }

  function resizeActiveTrackCharts(track) {
    setTimeout(() => {
      if (track === 'regression') {
        Object.values(State.regression.charts).forEach(c => c && c.resize());
      } else if (track === 'classification') {
        Object.values(State.classification.charts).forEach(c => c && c.resize());
      } else if (track === 'clustering') {
        Object.values(State.clustering.charts).forEach(c => c && c.resize());
      }
    }, 60);
  }

  // =========================================================================
  // HEALTH CHECK
  // =========================================================================
  async function checkHealth() {
    const dot = document.getElementById('health-dot');
    const text = document.getElementById('health-text');
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      State.health = data;

      if (data.status === 'ok') {
        dot.classList.add('ok');
        text.textContent = 'All Systems Live';
      } else {
        dot.classList.remove('ok');
        text.textContent = 'Partial Systems';
      }

      // Check track-level ready banners
      const regBanner = document.getElementById('reg-not-ready');
      if (regBanner) regBanner.style.display = data.regression_ready ? 'none' : 'flex';

      const clfBanner = document.getElementById('clf-not-ready');
      if (clfBanner) clfBanner.style.display = data.classification_ready ? 'none' : 'flex';

      const clustBanner = document.getElementById('clust-not-ready');
      if (clustBanner) clustBanner.style.display = data.clustering_ready ? 'none' : 'flex';
    } catch (err) {
      console.warn('Health check failed:', err);
      if (dot) dot.classList.remove('ok');
      if (text) text.textContent = 'Backend Offline';
    }
  }

  // =========================================================================
  // TRACK 1: REGRESSION (DIAMOND PRICE PREDICTION)
  // =========================================================================
  async function initRegression() {
    try {
      const res = await fetch('/api/regression/info');
      const data = await res.json();
      if (data.status !== 'success') return;

      State.regression.info = data;
      populateRegressionControls(data);
      renderRegressionModelTable(data.models, data.best_model);
      initRegressionCharts(data);
      bindRegressionEvents();
    } catch (err) {
      console.error('Error initializing regression track:', err);
    }
  }

  function populateRegressionControls(data) {
    // Models dropdown
    const modelSelect = document.getElementById('reg-model-select');
    if (modelSelect) {
      modelSelect.innerHTML = '';
      (data.available_models || Object.keys(data.models)).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m === data.best_model ? `${m} (Best Model)` : m;
        if (m === data.best_model) opt.selected = true;
        modelSelect.appendChild(opt);
      });
    }

    // Cut Options
    const cutSelect = document.getElementById('reg-cut');
    if (cutSelect && data.categorical_info?.cut_options) {
      cutSelect.innerHTML = '';
      data.categorical_info.cut_options.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        if (c === 'Ideal') opt.selected = true;
        cutSelect.appendChild(opt);
      });
    }

    // Color Options
    const colorSelect = document.getElementById('reg-color');
    if (colorSelect && data.categorical_info?.color_options) {
      colorSelect.innerHTML = '';
      data.categorical_info.color_options.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        if (c === 'G') opt.selected = true;
        colorSelect.appendChild(opt);
      });
    }

    // Clarity Options
    const claritySelect = document.getElementById('reg-clarity');
    if (claritySelect && data.categorical_info?.clarity_options) {
      claritySelect.innerHTML = '';
      data.categorical_info.clarity_options.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        if (c === 'VS2') opt.selected = true;
        claritySelect.appendChild(opt);
      });
    }
  }

  function renderRegressionModelTable(models, bestModel) {
    const wrap = document.getElementById('reg-model-table-wrap');
    if (!wrap) return;

    let html = `
      <div class="data-table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Model Name</th>
              <th>R² Score</th>
              <th>RMSE ($)</th>
              <th>MAE ($)</th>
              <th>Evaluation Status</th>
            </tr>
          </thead>
          <tbody>
    `;

    Object.entries(models).forEach(([name, metrics]) => {
      const isBest = name === bestModel;
      html += `
        <tr class="${isBest ? 'highlight' : ''}">
          <td><strong>${name}</strong>${isBest ? ' <span class="best-badge">Best Model</span>' : ''}</td>
          <td>${metrics.r2.toFixed(4)}</td>
          <td>$${metrics.rmse.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>$${metrics.mae.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${isBest ? 'Production Baseline' : 'Evaluated'}</td>
        </tr>
      `;
    });

    html += `
          </tbody>
        </table>
      </div>
    `;
    wrap.innerHTML = html;
  }

  function initRegressionCharts(data) {
    // 1. Actual vs Predicted Scatter
    const scatterCanvas = document.getElementById('reg-actual-pred-chart');
    if (scatterCanvas && data.actual_vs_pred) {
      const actuals = data.actual_vs_pred.actual || [];
      const preds = data.actual_vs_pred.predicted || [];

      const points = [];
      for (let i = 0; i < actuals.length; i++) {
        points.push({ x: actuals[i], y: preds[i] });
      }

      const minVal = 0;
      const maxVal = 20000;

      State.regression.charts.actualPred = new Chart(scatterCanvas, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Predictions',
              data: points,
              backgroundColor: 'rgba(59, 130, 246, 0.55)',
              borderColor: 'transparent',
              pointRadius: 3,
              pointHoverRadius: 5
            },
            {
              label: 'Ideal Reference (y = x)',
              data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
              type: 'line',
              borderColor: 'rgba(239, 68, 68, 0.85)',
              borderWidth: 2,
              borderDash: [5, 5],
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          ...CHART_DEFAULTS,
          scales: {
            x: {
              ...CHART_DEFAULTS.scales.x,
              title: { display: true, text: 'Actual Price ($)', color: '#94a3b8' }
            },
            y: {
              ...CHART_DEFAULTS.scales.y,
              title: { display: true, text: 'Predicted Price ($)', color: '#94a3b8' }
            }
          }
        }
      });
    }

    // 2. Residual Distribution Chart
    const resCanvas = document.getElementById('reg-residual-chart');
    if (resCanvas && data.actual_vs_pred?.residuals) {
      const resids = data.actual_vs_pred.residuals;
      // Bin residuals into 15 intervals
      const binCount = 15;
      const minRes = -2500;
      const maxRes = 2500;
      const step = (maxRes - minRes) / binCount;
      const bins = Array(binCount).fill(0);
      const labels = [];

      for (let i = 0; i < binCount; i++) {
        const start = minRes + i * step;
        const end = start + step;
        labels.push(`${Math.round(start)} to ${Math.round(end)}`);
      }

      resids.forEach(r => {
        if (r >= minRes && r <= maxRes) {
          const idx = Math.min(binCount - 1, Math.floor((r - minRes) / step));
          bins[idx]++;
        }
      });

      State.regression.charts.residual = new Chart(resCanvas, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Residual Count',
            data: bins,
            backgroundColor: 'rgba(16, 185, 129, 0.65)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4
          }]
        },
        options: {
          ...CHART_DEFAULTS,
          scales: {
            x: {
              ...CHART_DEFAULTS.scales.x,
              ticks: { display: false },
              title: { display: true, text: 'Residual Range ($) [Centred around 0]', color: '#94a3b8' }
            },
            y: {
              ...CHART_DEFAULTS.scales.y,
              title: { display: true, text: 'Frequency', color: '#94a3b8' }
            }
          }
        }
      });
    }

    // 3. Feature Importance Chart
    const featCanvas = document.getElementById('reg-feat-imp-chart');
    if (featCanvas && data.feature_importance) {
      const feats = data.feature_importance.slice(0, 10);
      State.regression.charts.featImp = new Chart(featCanvas, {
        type: 'bar',
        data: {
          labels: feats.map(f => f.feature),
          datasets: [{
            label: 'Relative Importance',
            data: feats.map(f => (f.importance * 100).toFixed(2)),
            backgroundColor: 'rgba(59, 130, 246, 0.7)',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4
          }]
        },
        options: {
          ...CHART_DEFAULTS,
          indexAxis: 'y',
          scales: {
            x: {
              ...CHART_DEFAULTS.scales.x,
              title: { display: true, text: 'Importance (%)', color: '#94a3b8' }
            },
            y: { ...CHART_DEFAULTS.scales.y }
          }
        }
      });
    }
  }

  function bindRegressionEvents() {
    const form = document.getElementById('reg-form');
    const resetBtn = document.getElementById('reg-reset-btn');
    const loadingElem = document.getElementById('reg-loading');
    const contentElem = document.getElementById('reg-result-content');
    const statusPill = document.getElementById('reg-status-pill');

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        form.reset();
        statusPill.textContent = 'Awaiting input';
        statusPill.className = 'status-pill';
        contentElem.innerHTML = `
          <div class="result-placeholder">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2a10 10 0 100 20A10 10 0 0012 2z"/><path d="M12 8v4l3 3"/></svg>
            <p>Enter diamond attributes and click <strong>Predict Price</strong></p>
          </div>
        `;
      });
    }

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
          model: document.getElementById('reg-model-select').value,
          carat: parseFloat(document.getElementById('reg-carat').value),
          depth: parseFloat(document.getElementById('reg-depth').value),
          table: parseFloat(document.getElementById('reg-table').value),
          x: parseFloat(document.getElementById('reg-x').value),
          y: parseFloat(document.getElementById('reg-y').value),
          z: parseFloat(document.getElementById('reg-z').value),
          cut: document.getElementById('reg-cut').value,
          color: document.getElementById('reg-color').value,
          clarity: document.getElementById('reg-clarity').value,
        };

        loadingElem.classList.remove('hidden');
        contentElem.classList.add('hidden');

        try {
          const res = await fetch('/api/regression/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const result = await res.json();
          loadingElem.classList.add('hidden');
          contentElem.classList.remove('hidden');

          if (result.status === 'success') {
            statusPill.textContent = 'Prediction Complete';
            statusPill.className = 'status-pill ready';

            const price = result.predicted_price;
            const pricePerCarat = (price / payload.carat).toFixed(2);
            const volume = result.inputs.volume;

            contentElem.innerHTML = `
              <div class="output-hero">
                <span class="output-label">Estimated Market Valuation</span>
                <div class="output-value-large">${result.formatted_price}</div>
                <div class="output-sub">Model: <strong>${result.model_used}</strong> · Deterministic Inference</div>
              </div>

              <div class="metrics-row">
                <div class="metric-box">
                  <span class="label">Price / Carat</span>
                  <span class="val">$${Number(pricePerCarat).toLocaleString()}</span>
                </div>
                <div class="metric-box">
                  <span class="label">Calculated Volume</span>
                  <span class="val">${volume} mm³</span>
                </div>
                <div class="metric-box">
                  <span class="label">Carat Weight</span>
                  <span class="val">${payload.carat} ct</span>
                </div>
              </div>

              <div class="data-table-wrap">
                <table class="data-table">
                  <tbody>
                    <tr><td>Cut Grade</td><td><strong>${payload.cut}</strong></td></tr>
                    <tr><td>Color Grade</td><td><strong>${payload.color}</strong></td></tr>
                    <tr><td>Clarity Grade</td><td><strong>${payload.clarity}</strong></td></tr>
                    <tr><td>Dimensions (x × y × z)</td><td>${payload.x} × ${payload.y} × ${payload.z} mm</td></tr>
                    <tr><td>Depth / Table Ratio</td><td>${payload.depth}% / ${payload.table}%</td></tr>
                  </tbody>
                </table>
              </div>
            `;
          } else {
            statusPill.textContent = 'Error';
            statusPill.className = 'status-pill fraud';
            contentElem.innerHTML = `
              <div class="output-hero fraud">
                <span class="output-label">Inference Error</span>
                <p style="color:var(--color-rose); margin-top:6px; font-size:0.86rem;">${result.message || 'Validation failed.'}</p>
              </div>
            `;
          }
        } catch (err) {
          loadingElem.classList.add('hidden');
          contentElem.classList.remove('hidden');
          statusPill.textContent = 'Network Error';
          statusPill.className = 'status-pill fraud';
          contentElem.innerHTML = `<p style="color:var(--color-rose); padding:16px;">Failed to contact server.</p>`;
        }
      });
    }
  }

  // =========================================================================
  // TRACK 2: CLASSIFICATION (FRAUD DETECTION)
  // =========================================================================
  async function initClassification() {
    try {
      const res = await fetch('/api/classification/info');
      const data = await res.json();
      if (data.status !== 'success') return;

      State.classification.info = data;
      populateClassificationControls(data);
      renderClassificationModelTable(data.models, data.best_model);
      initClassificationCharts(data);
      bindClassificationEvents();
    } catch (err) {
      console.error('Error initializing classification track:', err);
    }
  }

  function populateClassificationControls(data) {
    // Model Select
    const modelSelect = document.getElementById('clf-model-select');
    if (modelSelect) {
      modelSelect.innerHTML = '';
      (data.available_models || Object.keys(data.models)).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m === data.best_model ? `${m} (Best Model)` : m;
        if (m === data.best_model) opt.selected = true;
        modelSelect.appendChild(opt);
      });
    }

    // Chart Model Select
    const chartModelSelect = document.getElementById('clf-chart-model-select');
    if (chartModelSelect) {
      chartModelSelect.innerHTML = '';
      Object.keys(data.models).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === State.classification.selectedChartModel) opt.selected = true;
        chartModelSelect.appendChild(opt);
      });
    }

    // Categorical selects
    const catOpts = data.cat_options || {};
    const map = {
      'clf-merchant_category': 'merchant_category',
      'clf-device_type': 'device_type',
      'clf-transaction_country': 'transaction_country',
      'clf-transaction_type': 'transaction_type',
      'clf-geo_location_region': 'geo_location_region'
    };

    Object.entries(map).forEach(([elemId, key]) => {
      const select = document.getElementById(elemId);
      if (select && catOpts[key]) {
        select.innerHTML = '';
        catOpts[key].forEach(val => {
          const opt = document.createElement('option');
          opt.value = val;
          opt.textContent = val.replace(/_/g, ' ');
          select.appendChild(opt);
        });
      }
    });
  }

  function renderClassificationModelTable(models, bestModel) {
    const wrap = document.getElementById('clf-model-table-wrap');
    if (!wrap) return;

    let html = `
      <div class="data-table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Classifier</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
    `;

    Object.entries(models).forEach(([name, metrics]) => {
      const isBest = name === bestModel;
      html += `
        <tr class="${isBest ? 'highlight' : ''}">
          <td><strong>${name}</strong>${isBest ? ' <span class="best-badge">Best Model</span>' : ''}</td>
          <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
          <td>${(metrics.precision * 100).toFixed(2)}%</td>
          <td>${(metrics.recall * 100).toFixed(2)}%</td>
          <td>${(metrics.f1 * 100).toFixed(2)}%</td>
          <td>${metrics.roc_auc ? metrics.roc_auc.toFixed(4) : 'N/A'}</td>
        </tr>
      `;
    });

    html += `
          </tbody>
        </table>
      </div>
    `;
    wrap.innerHTML = html;
  }

  function initClassificationCharts(data) {
    updateConfusionChart(data, State.classification.selectedChartModel);
    updateRocChart(data, State.classification.selectedChartModel);
    initMetricsComparisonChart(data);
  }

  function updateConfusionChart(data, modelName) {
    const wrap = document.getElementById('clf-confusion-wrap');
    if (!wrap) return;

    const m = data.models[modelName];
    if (!m || !m.confusion_matrix) {
      wrap.innerHTML = '<p class="text-muted" style="text-align:center;padding:40px;">No confusion matrix data available for this model.</p>';
      return;
    }

    const cm = m.confusion_matrix;
    const tn = Number(cm[0][0]);
    const fp = Number(cm[0][1]);
    const fn = Number(cm[1][0]);
    const tp = Number(cm[1][1]);

    const totalLegit = tn + fp;
    const totalFraud = fn + tp;
    const totalAll = totalLegit + totalFraud;

    const tnRate = totalLegit > 0 ? ((tn / totalLegit) * 100).toFixed(1) : '0.0';
    const fpRate = totalLegit > 0 ? ((fp / totalLegit) * 100).toFixed(1) : '0.0';
    const fnRate = totalFraud > 0 ? ((fn / totalFraud) * 100).toFixed(1) : '0.0';
    const tpRate = totalFraud > 0 ? ((tp / totalFraud) * 100).toFixed(1) : '0.0';

    const accuracy = totalAll > 0 ? (((tn + tp) / totalAll) * 100).toFixed(2) : '0.0';
    const recall = (tp + fn) > 0 ? ((tp / (tp + fn)) * 100).toFixed(2) : '0.0';
    const specificity = (tn + fp) > 0 ? ((tn / (tn + fp)) * 100).toFixed(2) : '0.0';
    const precision = (tp + fp) > 0 ? ((tp / (tp + fp)) * 100).toFixed(2) : '0.0';

    wrap.innerHTML = `
      <div class="cm-container">
        <div class="cm-axis-top">
          <div></div>
          <div>Predicted Legit (0)</div>
          <div>Predicted Fraud (1)</div>
        </div>
        <div class="cm-body">
          <div class="cm-row">
            <div class="cm-row-label">Actual<br>Legit (0)</div>
            <div class="cm-cell tn" title="True Negative: Correctly flagged as legitimate">
              <div class="cm-cell-hdr">
                <span class="cm-tag">True Negative</span>
                <span class="cm-rate">${tnRate}%</span>
              </div>
              <div class="cm-val">${tn.toLocaleString()}</div>
              <div class="cm-desc">Correctly classified legitimate</div>
            </div>
            <div class="cm-cell fp" title="False Positive: Type I error (false alarm)">
              <div class="cm-cell-hdr">
                <span class="cm-tag">False Positive</span>
                <span class="cm-rate">${fpRate}%</span>
              </div>
              <div class="cm-val">${fp.toLocaleString()}</div>
              <div class="cm-desc">Type I error (False alarm)</div>
            </div>
          </div>

          <div class="cm-row">
            <div class="cm-row-label">Actual<br>Fraud (1)</div>
            <div class="cm-cell fn" title="False Negative: Type II error (missed fraud)">
              <div class="cm-cell-hdr">
                <span class="cm-tag">False Negative</span>
                <span class="cm-rate">${fnRate}%</span>
              </div>
              <div class="cm-val">${fn.toLocaleString()}</div>
              <div class="cm-desc">Type II error (Missed fraud)</div>
            </div>
            <div class="cm-cell tp" title="True Positive: Correctly identified fraud">
              <div class="cm-cell-hdr">
                <span class="cm-tag">True Positive</span>
                <span class="cm-rate">${tpRate}%</span>
              </div>
              <div class="cm-val">${tp.toLocaleString()}</div>
              <div class="cm-desc">Correctly caught fraud</div>
            </div>
          </div>
        </div>

        <div class="cm-stats-strip">
          <div class="cm-stat-item">
            <span class="cm-stat-label">Accuracy</span>
            <span class="cm-stat-val">${accuracy}%</span>
          </div>
          <div class="cm-stat-item">
            <span class="cm-stat-label">Recall (Sens.)</span>
            <span class="cm-stat-val">${recall}%</span>
          </div>
          <div class="cm-stat-item">
            <span class="cm-stat-label">Specificity</span>
            <span class="cm-stat-val">${specificity}%</span>
          </div>
          <div class="cm-stat-item">
            <span class="cm-stat-label">Precision</span>
            <span class="cm-stat-val">${precision}%</span>
          </div>
        </div>
      </div>
    `;
  }

  function updateRocChart(data, modelName) {
    const canvas = document.getElementById('clf-roc-chart');
    if (!canvas) return;

    const m = data.models[modelName];
    let fpr = m?.roc_fpr ? [...m.roc_fpr] : [0, 1];
    let tpr = m?.roc_tpr ? [...m.roc_tpr] : [0, 1];

    if (fpr[0] > 0 || tpr[0] > 0) {
      fpr.unshift(0);
      tpr.unshift(0);
    }
    if (fpr[fpr.length - 1] < 1 || tpr[tpr.length - 1] < 1) {
      fpr.push(1);
      tpr.push(1);
    }

    const points = fpr.map((x, i) => ({ x: Number(x), y: Number(tpr[i] !== undefined ? tpr[i] : 1) }));

    if (State.classification.charts.roc) {
      State.classification.charts.roc.destroy();
    }

    State.classification.charts.roc = new Chart(canvas, {
      type: 'line',
      data: {
        datasets: [
          {
            label: `${modelName} (AUC = ${m?.roc_auc !== undefined ? Number(m.roc_auc).toFixed(4) : 'N/A'})`,
            data: points,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.18)',
            borderWidth: 2.5,
            fill: true,
            pointRadius: points.length < 25 ? 3 : 0,
            pointHoverRadius: 6,
            tension: 0.1,
            showLine: true
          },
          {
            label: 'Random Baseline (AUC = 0.5000)',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: 'rgba(148, 163, 184, 0.45)',
            borderDash: [5, 5],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            showLine: true
          }
        ]
      },
      options: {
        ...CHART_DEFAULTS,
        scales: {
          x: {
            ...CHART_DEFAULTS.scales.x,
            type: 'linear',
            min: 0,
            max: 1,
            ticks: {
              ...CHART_DEFAULTS.scales.x.ticks,
              stepSize: 0.2
            },
            title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#94a3b8' }
          },
          y: {
            ...CHART_DEFAULTS.scales.y,
            type: 'linear',
            min: 0,
            max: 1,
            ticks: {
              ...CHART_DEFAULTS.scales.y.ticks,
              stepSize: 0.2
            },
            title: { display: true, text: 'True Positive Rate (Recall)', color: '#94a3b8' }
          }
        }
      }
    });
  }

  function initMetricsComparisonChart(data) {
    const canvas = document.getElementById('clf-metrics-chart');
    if (!canvas) return;

    const names = Object.keys(data.models);
    const accuracies = names.map(n => (data.models[n].accuracy * 100).toFixed(1));
    const recalls = names.map(n => (data.models[n].recall * 100).toFixed(1));
    const precisions = names.map(n => (data.models[n].precision * 100).toFixed(1));
    const rocAucs = names.map(n => (data.models[n].roc_auc * 100).toFixed(1));

    State.classification.charts.metrics = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [
          { label: 'Accuracy (%)', data: accuracies, backgroundColor: 'rgba(59, 130, 246, 0.7)' },
          { label: 'Recall (%)', data: recalls, backgroundColor: 'rgba(239, 68, 68, 0.7)' },
          { label: 'Precision (%)', data: precisions, backgroundColor: 'rgba(245, 158, 11, 0.7)' },
          { label: 'ROC-AUC (%)', data: rocAucs, backgroundColor: 'rgba(16, 185, 129, 0.7)' }
        ]
      },
      options: {
        ...CHART_DEFAULTS,
        scales: {
          x: { ...CHART_DEFAULTS.scales.x },
          y: {
            ...CHART_DEFAULTS.scales.y,
            min: 0, max: 100,
            title: { display: true, text: 'Score (%)', color: '#94a3b8' }
          }
        }
      }
    });
  }

  function bindClassificationEvents() {
    const form = document.getElementById('clf-form');
    const resetBtn = document.getElementById('clf-reset-btn');
    const chartModelSelect = document.getElementById('clf-chart-model-select');
    const loadingElem = document.getElementById('clf-loading');
    const contentElem = document.getElementById('clf-result-content');
    const statusPill = document.getElementById('clf-status-pill');

    if (chartModelSelect) {
      chartModelSelect.addEventListener('change', (e) => {
        State.classification.selectedChartModel = e.target.value;
        if (State.classification.info) {
          updateConfusionChart(State.classification.info, e.target.value);
          updateRocChart(State.classification.info, e.target.value);
        }
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        form.reset();
        statusPill.textContent = 'Awaiting input';
        statusPill.className = 'status-pill';
        contentElem.innerHTML = `
          <div class="result-placeholder">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            <p>Enter transaction details and click <strong>Classify Transaction</strong></p>
          </div>
        `;
      });
    }

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
          model: document.getElementById('clf-model-select').value,
          transaction_amount: parseFloat(document.getElementById('clf-transaction_amount').value),
          avg_transaction_amount_30d_customer: parseFloat(document.getElementById('clf-avg_transaction_amount_30d_customer').value),
          customer_risk_score: parseFloat(document.getElementById('clf-customer_risk_score').value),
          account_age_days: parseInt(document.getElementById('clf-account_age_days').value),
          previous_chargebacks: parseInt(document.getElementById('clf-previous_chargebacks').value),
          customer_total_transactions_30d: parseInt(document.getElementById('clf-customer_total_transactions_30d').value),
          transaction_velocity_1h: parseInt(document.getElementById('clf-transaction_velocity_1h').value),
          transaction_velocity_24h: parseInt(document.getElementById('clf-transaction_velocity_24h').value),
          transaction_hour: parseInt(document.getElementById('clf-transaction_hour').value),
          transaction_day_of_week: parseInt(document.getElementById('clf-transaction_day_of_week').value),
          merchant_category: document.getElementById('clf-merchant_category').value,
          device_type: document.getElementById('clf-device_type').value,
          transaction_country: document.getElementById('clf-transaction_country').value,
          transaction_type: document.getElementById('clf-transaction_type').value,
          geo_location_region: document.getElementById('clf-geo_location_region').value,
          is_international: 0,
          is_high_risk_merchant_category: 0,
          is_weekend: 0
        };

        loadingElem.classList.remove('hidden');
        contentElem.classList.add('hidden');

        try {
          const res = await fetch('/api/classification/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const result = await res.json();
          loadingElem.classList.add('hidden');
          contentElem.classList.remove('hidden');

          if (result.status === 'success') {
            const pred = result.prediction;
            const isFraud = pred.class === 1;

            statusPill.textContent = isFraud ? 'Fraud Alert' : 'Legitimate';
            statusPill.className = `status-pill ${isFraud ? 'fraud' : 'legit'}`;

            const amountRatio = (payload.transaction_amount / (payload.avg_transaction_amount_30d_customer + 1)).toFixed(2);

            contentElem.innerHTML = `
              <div class="output-hero ${isFraud ? 'fraud' : 'legit'}">
                <span class="output-label">Model Verdict</span>
                <div class="output-value-large">${pred.label}</div>
                <div class="output-sub">Model: <strong>${result.model_used}</strong> · Risk Level: <strong>${pred.risk_level}</strong></div>
              </div>

              <div class="metrics-row">
                <div class="metric-box">
                  <span class="label">Fraud Probability</span>
                  <span class="val" style="color:${isFraud ? 'var(--color-rose)' : 'var(--text-main)'}">${pred.fraud_probability}%</span>
                </div>
                <div class="metric-box">
                  <span class="label">Legit Probability</span>
                  <span class="val">${pred.legitimate_probability}%</span>
                </div>
                <div class="metric-box">
                  <span class="label">Amount Ratio</span>
                  <span class="val">${amountRatio}x</span>
                </div>
              </div>

              <div class="data-table-wrap">
                <table class="data-table">
                  <tbody>
                    <tr><td>Txn Amount</td><td>$${payload.transaction_amount.toFixed(2)}</td></tr>
                    <tr><td>30d Customer Avg</td><td>$${payload.avg_transaction_amount_30d_customer.toFixed(2)}</td></tr>
                    <tr><td>Previous Chargebacks</td><td><strong>${payload.previous_chargebacks}</strong></td></tr>
                    <tr><td>Customer Risk Score</td><td>${payload.customer_risk_score} / 100</td></tr>
                    <tr><td>Velocity (1h / 24h)</td><td>${payload.transaction_velocity_1h} / ${payload.transaction_velocity_24h}</td></tr>
                    <tr><td>Merchant & Device</td><td>${payload.merchant_category} · ${payload.device_type}</td></tr>
                  </tbody>
                </table>
              </div>
            `;
          } else {
            statusPill.textContent = 'Error';
            statusPill.className = 'status-pill fraud';
            contentElem.innerHTML = `
              <div class="output-hero fraud">
                <span class="output-label">Classification Error</span>
                <p style="color:var(--color-rose); margin-top:6px;">${result.message || 'Check features.'}</p>
              </div>
            `;
          }
        } catch (err) {
          loadingElem.classList.add('hidden');
          contentElem.classList.remove('hidden');
          statusPill.textContent = 'Network Error';
          statusPill.className = 'status-pill fraud';
          contentElem.innerHTML = `<p style="color:var(--color-rose); padding:16px;">Failed to reach inference server.</p>`;
        }
      });
    }
  }

  // =========================================================================
  // TRACK 3: CLUSTERING (AMAZON PRODUCT CATALOG SEGMENTATION)
  // =========================================================================
  async function initClustering() {
    try {
      const res = await fetch('/api/clustering/overview');
      const data = await res.json();
      if (data.status !== 'success') return;

      State.clustering.overview = data;
      renderClusteringKPIs(data);
      renderClusterProfileCards(data.clusters);
      initClusteringCharts(data);
      loadPcaPoints();
      initCatalogExplorer(data.dataset.categories);
      bindClusteringEvents();
    } catch (err) {
      console.error('Error initializing clustering track:', err);
    }
  }

  function renderClusteringKPIs(data) {
    const bar = document.getElementById('clust-kpi-bar');
    if (!bar) return;

    bar.innerHTML = `
      <div class="kpi-item">
        <span class="label">Clean Records</span>
        <div class="val">${data.dataset.clean_records.toLocaleString()}</div>
        <div class="sub">Raw: ${data.dataset.total_raw.toLocaleString()} items</div>
      </div>
      <div class="kpi-item">
        <span class="label">Optimal Clusters</span>
        <div class="val">k = ${data.model.k}</div>
        <div class="sub">K-Means with k-means++</div>
      </div>
      <div class="kpi-item">
        <span class="label">Silhouette Score</span>
        <div class="val">${data.model.silhouette.toFixed(4)}</div>
        <div class="sub">Subsampled N=10,000</div>
      </div>
      <div class="kpi-item">
        <span class="label">Davies–Bouldin</span>
        <div class="val">${data.model.davies_bouldin.toFixed(4)}</div>
        <div class="sub">Separation ratio (lower is better)</div>
      </div>
    `;

    // Diagnostics row
    const diagRow = document.getElementById('clust-metrics-row');
    if (diagRow) {
      diagRow.innerHTML = `
        <div class="metric-box">
          <span class="label">Silhouette Score</span>
          <span class="val">${data.model.silhouette.toFixed(4)}</span>
        </div>
        <div class="metric-box">
          <span class="label">Davies-Bouldin Index</span>
          <span class="val">${data.model.davies_bouldin.toFixed(4)}</span>
        </div>
        <div class="metric-box">
          <span class="label">Calinski-Harabasz Index</span>
          <span class="val">${data.model.calinski_harabasz.toLocaleString()}</span>
        </div>
      `;
    }
  }

  function renderClusterProfileCards(clusters) {
    const container = document.getElementById('clust-profile-cards');
    if (!container) return;

    container.innerHTML = '';
    clusters.forEach(c => {
      const topCats = Object.entries(c.top_categories || {})
        .slice(0, 3)
        .map(([name, pct]) => `${name} (${pct}%)`)
        .join(', ');

      const card = document.createElement('div');
      card.className = `cluster-card c${c.id}`;
      card.innerHTML = `
        <div class="c-tag">Cluster ${c.id} · ${c.tag}</div>
        <div class="c-title">${c.name}</div>
        <div class="c-stats-grid">
          <div class="c-stat-row">
            <span class="lbl">Catalog Share</span>
            <span class="v">${c.share}% (${c.count.toLocaleString()})</span>
          </div>
          <div class="c-stat-row">
            <span class="lbl">Median Price</span>
            <span class="v">$${c.median_price.toFixed(2)}</span>
          </div>
          <div class="c-stat-row">
            <span class="lbl">Median Reviews</span>
            <span class="v">${c.median_reviews.toLocaleString()}</span>
          </div>
          <div class="c-stat-row">
            <span class="lbl">Avg. Monthly Sold</span>
            <span class="v">${c.mean_bought.toLocaleString()}</span>
          </div>
          <div class="c-stat-row">
            <span class="lbl">Discount Share</span>
            <span class="v">${c.discount_rate}%</span>
          </div>
          <div class="c-stat-row">
            <span class="lbl">Rated Share</span>
            <span class="v">${c.rated_rate}%</span>
          </div>
        </div>
        <div class="c-cats"><strong>Leading categories:</strong> ${topCats || 'Varied across catalog'}</div>
      `;
      container.appendChild(card);
    });
  }

  function initClusteringCharts(data) {
    // 1. Cluster Size Distribution (Doughnut)
    const sizeCanvas = document.getElementById('clust-size-chart');
    if (sizeCanvas) {
      State.clustering.charts.size = new Chart(sizeCanvas, {
        type: 'doughnut',
        data: {
          labels: data.clusters.map(c => `Cluster ${c.id}: ${c.tag}`),
          datasets: [{
            data: data.clusters.map(c => c.count),
            backgroundColor: [CLUSTER_COLORS[0], CLUSTER_COLORS[1], CLUSTER_COLORS[2], CLUSTER_COLORS[3]],
            borderColor: '#121927',
            borderWidth: 2
          }]
        },
        options: {
          ...CHART_DEFAULTS,
          cutout: '62%'
        }
      });
    }

    // 2. Feature Profile by Cluster (Grouped Bar Chart)
    const profileCanvas = document.getElementById('clust-profile-chart');
    if (profileCanvas) {
      const clusters = data.clusters;
      State.clustering.charts.profile = new Chart(profileCanvas, {
        type: 'bar',
        data: {
          labels: ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'],
          datasets: [
            {
              label: 'Median Price ($)',
              data: clusters.map(c => c.median_price),
              backgroundColor: 'rgba(59, 130, 246, 0.7)'
            },
            {
              label: 'Discount Rate (%)',
              data: clusters.map(c => c.discount_rate),
              backgroundColor: 'rgba(16, 185, 129, 0.7)'
            },
            {
              label: 'Rated Share (%)',
              data: clusters.map(c => c.rated_rate),
              backgroundColor: 'rgba(245, 158, 11, 0.7)'
            }
          ]
        },
        options: {
          ...CHART_DEFAULTS,
          scales: {
            x: { ...CHART_DEFAULTS.scales.x },
            y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Metric Magnitude', color: '#94a3b8' } }
          }
        }
      });
    }

    // 3. Elbow Chart
    initElbowChart(data.elbow_data, data.selected_k);
  }

  function initElbowChart(elbowData, selectedK) {
    const canvas = document.getElementById('clust-elbow-chart');
    if (!canvas || !elbowData) return;

    renderElbowMetric(canvas, elbowData, State.clustering.activeElbowMetric, selectedK);

    // Toggle button events
    const toggleBtns = document.querySelectorAll('.chart-toggle-group .toggle-btn');
    toggleBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        toggleBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        State.clustering.activeElbowMetric = btn.dataset.elbow;
        renderElbowMetric(canvas, elbowData, State.clustering.activeElbowMetric, selectedK);
      });
    });
  }

  function renderElbowMetric(canvas, elbowData, metric, selectedK) {
    if (State.clustering.charts.elbow) {
      State.clustering.charts.elbow.destroy();
    }

    const metricKey = (metric === 'db' || metric === 'davies_bouldin') ? 'davies_bouldin' : metric;
    const labels = elbowData.map(d => `k = ${d.k}`);
    const values = elbowData.map(d => {
      const val = d[metricKey] !== undefined ? d[metricKey] : d[metric];
      return val !== undefined ? Number(val) : 0;
    });

    const metricTitles = {
      inertia: 'Inertia (Within-Cluster Sum of Squares)',
      silhouette: 'Silhouette Score',
      db: 'Davies–Bouldin Index',
      davies_bouldin: 'Davies–Bouldin Index'
    };

    const metricDescriptions = {
      inertia: 'Within-cluster sum of squares vs. k. Optimal k=4 selected at the elbow bend.',
      silhouette: 'Mean Silhouette score vs. k. Higher score indicates better cluster separation and cohesion.',
      db: 'Davies–Bouldin Index vs. k. Lower score indicates tighter and better separated clusters.',
      davies_bouldin: 'Davies–Bouldin Index vs. k. Lower score indicates tighter and better separated clusters.'
    };

    // Update card subtitle if present
    const card = canvas.closest('.card');
    if (card) {
      const desc = card.querySelector('.card-desc');
      if (desc && metricDescriptions[metricKey]) {
        desc.textContent = metricDescriptions[metricKey];
      }
    }

    const pointBg = elbowData.map(d => d.k === selectedK ? '#ef4444' : '#3b82f6');
    const pointRadii = elbowData.map(d => d.k === selectedK ? 8 : 4);

    State.clustering.charts.elbow = new Chart(canvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: metricTitles[metricKey] || metric,
          data: values,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.12)',
          pointBackgroundColor: pointBg,
          pointBorderColor: '#ffffff',
          pointRadius: pointRadii,
          pointHoverRadius: 9,
          borderWidth: 2.5,
          fill: true,
          tension: 0.25
        }]
      },
      options: {
        ...CHART_DEFAULTS,
        scales: {
          x: {
            ...CHART_DEFAULTS.scales.x,
            title: { display: true, text: 'Number of Clusters (k)', color: '#94a3b8' }
          },
          y: {
            ...CHART_DEFAULTS.scales.y,
            title: { display: true, text: metricTitles[metricKey] || metric, color: '#94a3b8' }
          }
        }
      }
    });
  }

  async function loadPcaPoints() {
    const canvas = document.getElementById('clust-pca-chart');
    if (!canvas) return;

    try {
      const res = await fetch('/api/clustering/pca_points');
      const data = await res.json();
      if (data.status !== 'success') return;

      State.clustering.pcaData = data;

      // Group points by cluster
      const clusters = [0, 1, 2, 3];
      const datasets = clusters.map(cId => {
        const clusterPoints = data.points
          .filter(p => p.cluster === cId)
          .map(p => ({
            x: p.x,
            y: p.y,
            title: p.title,
            price: p.price,
            category: p.category,
            stars: p.stars,
            reviews: p.reviews
          }));

        return {
          label: `Cluster ${cId}`,
          data: clusterPoints,
          backgroundColor: CLUSTER_COLORS[cId],
          pointRadius: 4,
          pointHoverRadius: 7
        };
      });

      // Dataset 4: User Classified Product Marker
      datasets.push({
        label: '⭐ Your Product',
        data: State.clustering.userPoint ? [State.clustering.userPoint] : [],
        backgroundColor: '#f59e0b',
        borderColor: '#ffffff',
        borderWidth: 2,
        pointStyle: 'star',
        pointRadius: 10,
        pointHoverRadius: 13
      });

      State.clustering.charts.pca = new Chart(canvas, {
        type: 'scatter',
        data: { datasets },
        options: {
          ...CHART_DEFAULTS,
          plugins: {
            ...CHART_DEFAULTS.plugins,
            tooltip: {
              ...CHART_DEFAULTS.plugins.tooltip,
              callbacks: {
                label: function (ctx) {
                  const p = ctx.raw;
                  if (ctx.datasetIndex === 4) {
                    return `⭐ Classified Product [PC1: ${p.x}, PC2: ${p.y}]`;
                  }
                  return [
                    `${p.title || 'Product'}`,
                    `Price: $${p.price} · Category: ${p.category}`,
                    `Rating: ⭐ ${p.stars} (${p.reviews} reviews)`
                  ];
                }
              }
            }
          },
          scales: {
            x: {
              ...CHART_DEFAULTS.scales.x,
              title: { display: true, text: `PC1 (${data.pc1_var}% Variance)`, color: '#94a3b8' }
            },
            y: {
              ...CHART_DEFAULTS.scales.y,
              title: { display: true, text: `PC2 (${data.pc2_var}% Variance)`, color: '#94a3b8' }
            }
          }
        }
      });
    } catch (err) {
      console.error('Failed to load PCA points:', err);
    }
  }

  function bindClusteringEvents() {
    const form = document.getElementById('clust-form');
    const resetBtn = document.getElementById('clust-reset-btn');
    const loadingElem = document.getElementById('clust-loading');
    const contentElem = document.getElementById('clust-result-content');
    const statusPill = document.getElementById('clust-status-pill');

    // Quick Presets
    const presets = {
      viral: { price: 19.99, listPrice: 29.99, stars: 4.5, reviews: 640, bought: 150 },
      discount: { price: 34.50, listPrice: 59.99, stars: 4.2, reviews: 120, bought: 45 },
      standard: { price: 42.00, listPrice: 0.00, stars: 4.3, reviews: 15, bought: 8 },
      unrated: { price: 149.99, listPrice: 0.00, stars: 0.0, reviews: 0, bought: 0 }
    };

    document.querySelectorAll('.preset-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const key = chip.dataset.preset;
        const p = presets[key];
        if (!p) return;

        document.getElementById('clust-price').value = p.price;
        document.getElementById('clust-listprice').value = p.listPrice;
        document.getElementById('clust-stars').value = p.stars;
        document.getElementById('clust-reviews').value = p.reviews;
        document.getElementById('clust-bought').value = p.bought;

        // Auto trigger submit
        if (form) form.dispatchEvent(new Event('submit'));
      });
    });

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        form.reset();
        statusPill.textContent = 'Awaiting input';
        statusPill.className = 'status-pill';
        contentElem.innerHTML = `
          <div class="result-placeholder">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/></svg>
            <p>Enter product attributes and click <strong>Classify Product</strong></p>
          </div>
        `;
      });
    }

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
          price: parseFloat(document.getElementById('clust-price').value),
          listPrice: parseFloat(document.getElementById('clust-listprice').value || 0),
          stars: parseFloat(document.getElementById('clust-stars').value || 0),
          reviews: parseInt(document.getElementById('clust-reviews').value || 0),
          boughtInLastMonth: parseInt(document.getElementById('clust-bought').value || 0)
        };

        loadingElem.classList.remove('hidden');
        contentElem.classList.add('hidden');

        try {
          const res = await fetch('/api/clustering/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const result = await res.json();
          loadingElem.classList.add('hidden');
          contentElem.classList.remove('hidden');

          if (result.status === 'success') {
            const pred = result.prediction;
            statusPill.textContent = `Assigned: Cluster ${pred.cluster}`;
            statusPill.className = 'status-pill ready';

            // Distance rows
            const distRows = pred.centroid_distances.map(d => `
              <tr class="${d.is_nearest ? 'highlight' : ''}">
                <td>Cluster ${d.cluster}: ${d.name}</td>
                <td><strong>${d.distance}</strong>${d.is_nearest ? ' <span class="best-badge">Nearest Centroid</span>' : ''}</td>
              </tr>
            `).join('');

            contentElem.innerHTML = `
              <div class="output-hero" style="border-left: 4px solid ${pred.color};">
                <span class="output-label">Predicted Cluster Segment</span>
                <div class="output-value-large" style="color: ${pred.color}; font-size:1.45rem;">
                  Cluster ${pred.cluster} · ${pred.name}
                </div>
                <div class="output-sub">Tag: <strong>${pred.tag}</strong> · Model Confidence: <strong>${pred.confidence}%</strong></div>
              </div>

              <div class="metrics-row">
                <div class="metric-box">
                  <span class="label">Cluster Share</span>
                  <span class="val">${pred.profile.share}%</span>
                </div>
                <div class="metric-box">
                  <span class="label">Cluster Med. Price</span>
                  <span class="val">$${pred.profile.median_price.toFixed(2)}</span>
                </div>
                <div class="metric-box">
                  <span class="label">PCA Coordinates</span>
                  <span class="val">(${pred.pca_coords.x}, ${pred.pca_coords.y})</span>
                </div>
              </div>

              <div class="data-table-wrap">
                <table class="data-table">
                  <thead>
                    <tr><th>Cluster Centroid</th><th>Euclidean Distance</th></tr>
                  </thead>
                  <tbody>${distRows}</tbody>
                </table>
              </div>
            `;

            // Update user point on PCA chart
            State.clustering.userPoint = { x: pred.pca_coords.x, y: pred.pca_coords.y };
            if (State.clustering.charts.pca) {
              State.clustering.charts.pca.data.datasets[4].data = [State.clustering.userPoint];
              State.clustering.charts.pca.update();
            }
          } else {
            statusPill.textContent = 'Error';
            statusPill.className = 'status-pill fraud';
            contentElem.innerHTML = `<p style="color:var(--color-rose); padding:16px;">${result.message || 'Clustering failed.'}</p>`;
          }
        } catch (err) {
          loadingElem.classList.add('hidden');
          contentElem.classList.remove('hidden');
          statusPill.textContent = 'Network Error';
          statusPill.className = 'status-pill fraud';
          contentElem.innerHTML = `<p style="color:var(--color-rose); padding:16px;">Failed to reach clustering backend.</p>`;
        }
      });
    }
  }

  // =========================================================================
  // CATALOG EXPLORER (PAGEABLE & FILTERABLE)
  // =========================================================================
  function initCatalogExplorer(categories) {
    const catSelect = document.getElementById('catalog-cat');
    if (catSelect && categories) {
      catSelect.innerHTML = '<option value="All">All Categories</option>';
      categories.forEach(cat => {
        const opt = document.createElement('option');
        opt.value = cat;
        opt.textContent = cat;
        catSelect.appendChild(opt);
      });
    }

    const searchInput = document.getElementById('catalog-search');
    const clusterSelect = document.getElementById('catalog-cluster');
    const sortSelect = document.getElementById('catalog-sort');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');

    // Debounced Search
    let searchTimeout = null;
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
          State.clustering.catalog.searchQuery = e.target.value;
          State.clustering.catalog.page = 1;
          fetchCatalog();
        }, 280);
      });
    }

    if (catSelect) {
      catSelect.addEventListener('change', (e) => {
        State.clustering.catalog.category = e.target.value;
        State.clustering.catalog.page = 1;
        fetchCatalog();
      });
    }

    if (clusterSelect) {
      clusterSelect.addEventListener('change', (e) => {
        State.clustering.catalog.cluster = e.target.value;
        State.clustering.catalog.page = 1;
        fetchCatalog();
      });
    }

    if (sortSelect) {
      sortSelect.addEventListener('change', (e) => {
        State.clustering.catalog.sortBy = e.target.value;
        State.clustering.catalog.page = 1;
        fetchCatalog();
      });
    }

    if (btnPrev) {
      btnPrev.addEventListener('click', () => {
        if (State.clustering.catalog.page > 1) {
          State.clustering.catalog.page--;
          fetchCatalog();
        }
      });
    }

    if (btnNext) {
      btnNext.addEventListener('click', () => {
        if (State.clustering.catalog.page < State.clustering.catalog.totalPages) {
          State.clustering.catalog.page++;
          fetchCatalog();
        }
      });
    }

    // Initial catalog fetch
    fetchCatalog();
  }

  async function fetchCatalog() {
    const c = State.clustering.catalog;
    const grid = document.getElementById('product-grid');
    const countElem = document.getElementById('catalog-count');
    const pageInfo = document.getElementById('catalog-page-info');
    const indicator = document.getElementById('pagination-indicator');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');

    if (grid) {
      grid.innerHTML = '<div class="loading-state" style="grid-column: span 3;"><div class="spinner"></div><span>Querying Amazon catalog…</span></div>';
    }

    const params = new URLSearchParams({
      search: c.searchQuery,
      category: c.category,
      cluster: c.cluster,
      sort: c.sortBy,
      page: c.page,
      page_size: c.pageSize
    });

    try {
      const res = await fetch(`/api/clustering/catalog?${params.toString()}`);
      const data = await res.json();
      if (data.status !== 'success') return;

      c.totalPages = data.total_pages;

      if (countElem) countElem.textContent = `${data.total.toLocaleString()} products found`;
      if (pageInfo) pageInfo.textContent = `Page ${data.page} of ${data.total_pages}`;
      if (indicator) indicator.textContent = `${data.page} / ${data.total_pages}`;

      if (btnPrev) btnPrev.disabled = data.page <= 1;
      if (btnNext) btnNext.disabled = data.page >= data.total_pages;

      if (!grid) return;
      if (data.items.length === 0) {
        grid.innerHTML = '<div style="grid-column:span 3; padding:40px; text-align:center; color:var(--text-faint);">No products match the selected criteria.</div>';
        return;
      }

      grid.innerHTML = '';
      data.items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'product-card-item';
        card.innerHTML = `
          <div>
            <div class="p-cat">${item.category}</div>
            <div class="p-title" title="${item.title}">${item.title}</div>
            <div class="p-pricing">
              <span class="p-price">$${item.price.toFixed(2)}</span>
              ${item.has_discount ? `<span class="p-listprice">$${item.listPrice.toFixed(2)}</span><span class="p-discount-tag">Save $${(item.listPrice - item.price).toFixed(2)}</span>` : ''}
            </div>
          </div>
          <div class="p-footer">
            <span>⭐ ${item.stars > 0 ? item.stars.toFixed(1) : 'Unrated'} (${item.reviews.toLocaleString()})</span>
            <span class="p-cluster-tag" style="background-color: ${item.cluster_color}">Cluster ${item.cluster}</span>
          </div>
        `;
        grid.appendChild(card);
      });
    } catch (err) {
      console.error('Catalog fetch failed:', err);
      if (grid) grid.innerHTML = '<div style="color:var(--color-rose); padding:20px;">Failed to load catalog data.</div>';
    }
  }

  // =========================================================================
  // BOOTSTRAP APPLICATION
  // =========================================================================
  initNavigation();
  checkHealth();
  initRegression();
  initClassification();
  initClustering();

  console.log('[ML Capstone] All controllers initialized.');
});
