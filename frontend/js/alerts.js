/* ═══════════════════════════════════════════════════════════════
   AEROS — Alert Display Manager
   Renders NLP health advisory cards and the scrolling bottom ticker.
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  const SEVERITY_COLOR = {
    Low: '#00e400',
    Moderate: '#ffb800',
    High: '#ff7e00',
    Critical: '#ff3838',
  };

  const CATEGORY_COLOR = {
    'Good': '#00e400',
    'Satisfactory': '#9cff9c',
    'Moderate': '#ffff00',
    'Poor': '#ff7e00',
    'Very Poor': '#ff0000',
    'Severe': '#99004c',
    'Severe+': '#7e0023',
  };

  const POLLUTANT_SYMBOLS = {
    pm25: 'PM2.5', pm10: 'PM10', no2: 'NO₂',
    so2: 'SO₂', o3: 'O₃', co: 'CO',
  };

  class AlertPanel {
    constructor(listEl, tickerEl, tickerMetaEl) {
      this.list = listEl;
      this.ticker = tickerEl;
      this.tickerMeta = tickerMetaEl;
    }

    render(alerts) {
      if (!alerts || !this.list) return;
      const items = (alerts || []).slice(0, 6);
      if (!items.length) {
        this.list.innerHTML = '<div class="dim" style="font-size:12px">No advisories yet.</div>';
        return;
      }

      this.list.innerHTML = items.map((a, i) => this._card(a, i)).join('');
      this.list.querySelectorAll('.alert-card').forEach((el) => {
        el.addEventListener('click', () => el.classList.toggle('open'));
      });
    }

    _rich(text) {
      // Escape HTML, then render **bold** markers from advisory text.
      const esc = Utils.esc(text || '');
      return esc.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    }

    _card(a, i) {
      // a.category is a string label ("Moderate"), not a number.
      const catColor = CATEGORY_COLOR[a.category] || '#ffb800';
      const sevColor = (a.severity && SEVERITY_COLOR[a.severity.level]) || catColor;
      const id = a.id || i;
      const domSym = POLLUTANT_SYMBOLS[a.dominant_pollutant] || (a.dominant_pollutant || '').toUpperCase();
      const genBadge = a.generator === 'llm-gemini' ? 'NLP · LLM'
        : a.generator === 'nlp-local' ? 'NLP · local'
        : a.generator ? Utils.esc(a.generator) : 'advisory';
      const metaBits = [
        a.station_name || '',
        a.category || '',
        domSym ? ('▸ ' + domSym) : '',
        (a.peak_pm25 != null ? ('PM2.5 ' + a.peak_pm25) : ''),
        (a.aisi != null ? ('AISI ' + a.aisi) : ''),
        (a.trend ? ('↗ ' + a.trend) : ''),
      ].filter(Boolean).join(' · ');

      const norm = (s) => (s || '').replace(/\*+/g, '').replace(/\s+/g, ' ').trim().toLowerCase();
      const summaryNorm = norm(a.summary);
      const sections = [
        ['Situation', a.sections?.situation],
        ['Health guidance', a.sections?.health_guidance],
        ['Duration', a.sections?.duration],
        ['GRAP', a.sections?.grap],
      ].filter(([, t]) => t && norm(t) !== summaryNorm);

      return `
        <div class="alert-card pop-in" style="--severe:${sevColor}" data-alert="${Utils.esc(id)}">
          <div class="alert-head">
            <span class="alert-title" style="color:${sevColor}">${Utils.esc(a.title || 'Advisory')}</span>
            <span class="alert-gen" title="generation engine">${Utils.esc(genBadge)}</span>
          </div>
          <div class="alert-time">${Utils.esc(metaBits)} · ${Utils.fmtTime(a.generated_at)}</div>
          <div class="alert-summary">${this._rich(a.summary || '')}</div>
          <div class="sections">
            ${sections.map(([t, body]) =>
              `<div class="section"><b>${Utils.esc(t)}</b><br>${this._rich(body)}</div>`).join('')}
          </div>
        </div>`;
    }

    renderTicker(alert, meta) {
      if (!this.ticker) return;
      const summary = (alert && alert.summary) ||
        'AEROS online — fetching live air quality telemetry…';
      this.ticker.innerHTML = `<span class="ticker-text">${Utils.esc(summary)}</span>`;
      if (this.tickerMeta) {
        this.tickerMeta.textContent = meta || '';
      }
    }
  }

  global.AlertPanel = AlertPanel;
})(window);
