/* ═══════════════════════════════════════════════════════════════
   AEROS — Plume Trajectory Visualizer
   Renders fire plumes and computes simple wind-streamline particles
   for the live map (source attribution + mass flux labels).
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  class PlumeOverlay {
    constructor(map) {
      this.map = map;
      this.geojson = { type: 'FeatureCollection', features: [] };
    }

    update(plume) {
      this.geojson = (plume && plume.geojson) || { type: 'FeatureCollection', features: [] };
      if (this.map && this.map.isStyleLoaded()) {
        const src = this.map.getSource('plumes');
        if (src) src.setData(this.geojson);
      }
    }

    /* UI helpers for the dashboard (source attribution chips) */
    renderArrivals(arrivals, containerId) {
      const el = document.getElementById(containerId);
      if (!el) return;
      if (!arrivals || !arrivals.length) {
        el.innerHTML = '<span class="dim">No incoming plumes detected in window</span>';
        return;
      }
      el.innerHTML = arrivals
        .slice(0, 6)
        .map((a) => {
          const color = a.estimated_contribution_pm25 > 10 ? '#DC2626'
            : a.estimated_contribution_pm25 > 4 ? '#D97706' : '#16A34A';
          return `<span class="chip" style="border-color:${color}">
            <b style="color:${color}">+${Number(a.estimated_contribution_pm25).toFixed(1)} µg/m³</b>
            arrive in ~${a.arrival_hours}h · ${a.distance_km} km
          </span>`;
        })
        .join('');
    }
  }

  global.PlumeOverlay = PlumeOverlay;
})(window);