/* ═══════════════════════════════════════════════════════════════
   AEROS — Utilities
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  const POLLUTANT_NAMES = {
    pm25: 'PM₂.₅',
    pm10: 'PM₁₀',
    no2:  'NO₂',
    so2:  'SO₂',
    o3:   'O₃',
    co:   'CO',
  };

  const AQI_CATEGORIES = [
    { label: 'Good',         min: 0,   max: 50,   color: '#00e400' },
    { label: 'Satisfactory', min: 51,  max: 100,  color: '#9cff9c' },
    { label: 'Moderate',     min: 101, max: 200,  color: '#ffff00' },
    { label: 'Poor',         min: 201, max: 300,  color: '#ff7e00' },
    { label: 'Very Poor',    min: 301, max: 400,  color: '#ff0000' },
    { label: 'Severe',       min: 401, max: 500,  color: '#99004c' },
    { label: 'Severe+',      min: 501, max: 999,  color: '#7e0023' },
  ];

  function aqiCategory(aqi) {
    const v = Number(aqi) || 0;
    for (const c of AQI_CATEGORIES) {
      if (v >= c.min && v <= c.max) return c;
    }
    return AQI_CATEGORIES[AQI_CATEGORIES.length - 1];
  }

  function aqiColor(aqi) {
    return aqiCategory(aqi).color;
  }

  // Display override: Moderate-yellow (#ffff00) reads poorly on light
  // chrome, so STATION markers/rows render it as black; pale Satisfactory
  // green (#9cff9c) is near-invisible on the light basemap, so it renders
  // as dark green. AQI math and legends elsewhere are untouched.
  const DISPLAY_COLOR_MAP = { '#ffff00': '#000000', '#9cff9c': '#007a00' };
  function stationDisplayColor(hex) {
    if (typeof hex === 'string') {
      const hit = DISPLAY_COLOR_MAP[hex.toLowerCase()];
      if (hit) return hit;
    }
    return hex;
  }

  function pollutantName(key) {
    return POLLUTANT_NAMES[key] || key.toUpperCase();
  }

  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts || {});
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    return res.json();
  }

  function wsUrl(path) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}${path}`;
  }

  function fmtTime(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function fmtDT(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString([], { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
  }

  function esc(text) {
    const div = document.createElement('div');
    div.textContent = String(text == null ? '' : text);
    return div.innerHTML;
  }

  function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
  }

  function debounce(fn, ms) {
    let t;
    return function () {
      clearTimeout(t);
      const args = arguments;
      t = setTimeout(() => fn.apply(null, args), ms);
    };
  }

  const Utils = {
    AQI_CATEGORIES,
    aqiCategory,
    aqiColor,
    stationDisplayColor,
    pollutantName,
    fetchJSON,
    wsUrl,
    fmtTime,
    fmtDT,
    esc,
    clamp,
    debounce,
  };

  global.Utils = Utils;
})(window);