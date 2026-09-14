/* ═══════════════════════════════════════════════════════════════
   AEROS — Station Forecast Timeseries (Chart.js)
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  class ForecastChart {
    constructor(canvasEl) {
      this.canvas = canvasEl;
      this.ctx = canvasEl.getContext('2d');
      this.forecast = null;
      this.history = [];   // observed past readings (flat {timestamp, pm25, pm10})
      this.pollutant = 'pm25';
      this.chart = null;
      this._build();
    }

    _build() {
      const grad = this.ctx.createLinearGradient(0, 0, 0, 240);
      grad.addColorStop(0, 'rgba(0,212,255,0.35)');
      grad.addColorStop(1, 'rgba(0,212,255,0.02)');

      this.chart = new Chart(this.ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 500 },
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(10,14,39,0.9)',
              borderColor: 'rgba(0,212,255,0.4)',
              borderWidth: 1,
              titleColor: '#fff',
              bodyColor: 'rgba(255,255,255,0.8)',
              callbacks: {
                label: (c) => `${c.dataset.label || 'Value'}: ${c.parsed.y}`,
              },
            },
          },
          scales: {
            x: {
              grid: { color: 'rgba(255,255,255,0.05)' },
              ticks: { color: 'rgba(255,255,255,0.45)', maxRotation: 0, maxTicksLimit: 10, font: { size: 10 } },
            },
            y: {
              grid: { color: 'rgba(255,255,255,0.06)' },
              ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 10 } },
            },
          },
        },
      });
      this.grad = grad;
    }

    setData(forecast) {
      this.forecast = forecast || null;
      this._render();
    }

    setHistory(rows) {
      // Keep the last 24 observed points with a usable value.
      this.history = (rows || [])
        .filter((r) => r && r.pm25 != null)
        .slice(-24);
      // Gap detection: pairs >3h apart are data gaps (sensor offline or
      // refresh missed). Count is surfaced in the chart notes; nulls in
      // the series render as visible line breaks (spanGaps:false).
      this.gapCount = 0;
      for (let i = 1; i < this.history.length; i++) {
        const a = new Date(this.history[i - 1].timestamp).getTime();
        const b = new Date(this.history[i].timestamp).getTime();
        if (isFinite(a) && isFinite(b) && (b - a) > 3 * 3600 * 1000) this.gapCount++;
      }
      this._render();
    }

    setPollutant(p) {
      this.pollutant = p;
      this._render();
    }

    _render() {
      if (!this.forecast) return;
      const f = this.forecast;
      const datasets = [];

      // Observed past (SQLite history) prepended to the forecast axis so
      // the chart reads as one continuous past → future timeline.
      const hist = this.history || [];
      const histLabels = hist.map((r) => Utils.fmtTime(r.timestamp));
      const pad = new Array(hist.length).fill(null);

      if (this.pollutant === 'aqi') {
        const labels = (f.timestamps || []).map((t) => Utils.fmtTime(t));
        datasets.push({
          label: 'NAQI',
          data: f.aqi || [],
          borderColor: (ctx) => this._pointColors(ctx),
          backgroundColor: (ctx) => this._pointColors(ctx, 0.45),
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.35,
        });
        this.chart.data.labels = labels;
      } else if (['pm25', 'pm10', 'no2', 'o3'].includes(this.pollutant)) {
        const key = this.pollutant;
        const base = f[key] || [];
        // Per-pollutant uncertainty envelope when exported
        // (<key>_upper/<key>_lower); PM2.5 legacy keys as fallback.
        const hi = f[`${key}_upper`] || (key === 'pm10' && f.pm10_upper) || (f.upper || []);
        const lo = f[`${key}_lower`] || (key === 'pm10' && f.pm10_lower) || (f.lower || []);

        const labels = histLabels.concat(
          (f.timestamps || []).map((t) => Utils.fmtTime(t))
        );

        if (hist.length) {
          datasets.push({
            label: 'Observed (past 24h)',
            data: hist.map((r) => (r[key] != null ? r[key] : null)),
            borderColor: '#000000',
            backgroundColor: 'rgba(0,0,0,0.06)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            spanGaps: false,
          });
        }
        datasets.push({
          label: `${Utils.pollutantName(key)} forecast`,
          data: pad.concat(base),
          borderColor: '#0070d1',
          backgroundColor: (ctx) => this._gradientFill(ctx),
          fill: hist.length ? false : true,
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.35,
        });
        datasets.push({
          label: 'Upper bound (90th)',
          data: pad.concat(hi),
          borderColor: 'rgba(0,112,209,0.35)',
          backgroundColor: 'rgba(0,112,209,0.08)',
          borderDash: [4, 4],
          pointRadius: 0,
          fill: '-1',
          tension: 0.35,
        });
        datasets.push({
          label: 'Lower bound (10th)',
          data: pad.concat(lo),
          borderColor: 'rgba(0,112,209,0.35)',
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
          tension: 0.35,
        });
        this.chart.data.labels = labels;
      } else {
        const labels = (f.timestamps || []).map((t) => Utils.fmtTime(t));
        datasets.push({
          label: Utils.pollutantName(this.pollutant),
          data: (f[this.pollutant] || []),
          borderColor: '#000000',
          backgroundColor: 'rgba(0,0,0,0.08)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
        });
        this.chart.data.labels = labels;
      }

      this.chart.data.datasets = datasets;
      this.chart.update();
    }

    _gradientFill(ctx) {
      const { chartArea } = ctx.chart;
      if (!chartArea) return 'rgba(0,112,209,0.10)';
      const g = ctx.chart.ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
      g.addColorStop(0, 'rgba(0,112,209,0.25)');
      g.addColorStop(1, 'rgba(0,112,209,0.0)');
      return g;
    }

    _pointColors(ctx, alpha = 1) {
      const { index } = ctx;
      const colors = this.forecast ? (this.forecast.colors || []) : [];
      if (!colors.length) return 'rgba(255,255,255,0.6)';
      const c = colors[index] || colors[0];
      return alpha < 1 ? hexA(c, alpha) : c;
    }
  }

  function hexA(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  global.ForecastChart = ForecastChart;
})(window);